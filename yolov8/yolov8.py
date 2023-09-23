import time
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from PIL import Image
import cupy as cp
import numpy as np
import tensorrt as trt

# from ultralytics.engine.results import Results
# from ultralytics.utils import DEFAULT_CFG, ROOT, ops

np.bool = bool

from yolov8.utils import (
    non_max_suppression, 
    scale_boxes, 
    Profile
)

# Load cuda kernels
path_curesize = Path(__file__).parent / "kernels" / "resize.cu"
assert path_curesize.exists(), f"Error: {str(path_curesize)} not found"
with open(path_curesize, 'r', encoding="utf-8") as reader:
    module = cp.RawModule(code=reader.read())
cuResizeKer = module.get_function("cuResize")


@dataclass(init=True)
class Binding:
    name: str = ""
    dtype: type = np.float32
    shape: tuple = (0, 0)
    data: np.ndarray = None
    ptr: int = None


class YOLOv8trt:


    """
    Class for the Yolo in TensorRT model
    """
    def __init__(self, model_path):
        """
        Initialize the model
        """
        # set allocation using unified memory in cupy (a.k.a. managed memory)
        # NOTE: THIS LINE WORKS ONLY WITH IGPU systems (like the jetson orin)
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool(cp.cuda.malloc_managed).malloc)

        # load model from .engine file
        logger = trt.Logger(trt.Logger.INFO)
        # self.__init_trt_plugins(logger)

        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
            self.model = runtime.deserialize_cuda_engine(f.read())  # read engine
        
        assert "imgsz" in metadata, "Missing 'imgsz' metadata"
        assert "batch" in metadata, "Missing 'batch' metadata"
        assert "task" in metadata, "Missing 'task' metadata"

        imgsz = metadata["imgsz"]
        
        self.device = "cuda:0"
        self.use_padding = True
        # NMS params
        self.conf = 0.25
        self.iou = 0.7
        self.agnostic_nms = False
        self.max_det = 300
        self.classes = None
        ############
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
        self.batch_size = metadata["batch"]
        self.task = metadata["task"]
        self.__call__ = self.predict
        
        # build the execution context
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False  # default updated below
        self.dynamic = False

        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            if self.model.binding_is_input(i):
                if -1 in tuple(self.model.get_binding_shape(i)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = cp.empty(shape, dtype=dtype)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data.ptr))
            # im = torch.empty(shape, dtype=torch.float, device=self.device)
            # self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        batch_size = self.bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        self.im_dtype = cp.float16 if self.fp16 else cp.float32

        self.profilers = [Profile(), Profile(), Profile()]
    

    def __init_trt_plugins(self, logger):
        """
        TODO: concatenate postprocess layers directly to the tensorrt model
        """
        trt.init_libnvinfer_plugins(logger, '')
        PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
        # ['BatchedNMSDynamic_TRT', 'BatchedNMS_TRT', 'EfficientNMS_Explicit_TF_TRT', 'EfficientNMS_Implicit_TF_TRT', 'EfficientNMS_ONNX_TRT', 'EfficientNMS_TRT', 'NMSDynamic_TRT', 'NMS_TRT']

        def get_trt_plugin(plugin_name):
            plugin = None
            for plugin_creator in PLUGIN_CREATORS:
                if plugin_creator.name == plugin_name:
                    lrelu_slope_field = trt.PluginField("neg_slope", np.array([0.1], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                    field_collection = trt.PluginFieldCollection([lrelu_slope_field])
                    plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
            return plugin

    def predict(self, img_orig: cp.ndarray):
        """
        Prediction step

        img: cupy array (B,H,W,C) with C = 3
        """
        with self.profilers[0]:
            img = self.preprocess_gpu(img_orig)

        with self.profilers[1]:
            y = self.forward(img)

        # newy = []
        # for i in y:
        #     newy.append(torch.as_tensor(i, device='cuda:0'))

        with self.profilers[2]:
            self.postprocess_detection(y, img.shape[2:4], img_orig.get())

        return y


    def forward(self, img):
        s = self.bindings['images'].shape
        assert img.shape == s, f"input size {img.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(img.data.ptr)
        # self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]


    def preprocess_gpu(self, im: cp.ndarray):
        """
        Very Fast Image Preprocessing

        Steps: 
        - resize and convert to float32/16
        - reshape from BHWC to BCHW
        - normalize image
        """
        # if isinstance(im, np.ndarray):
        #     mem = cp.cuda.UnownedMemory(im.ctypes.data, im.nbytes, im)
        #     mem_ptr = cp.cuda.MemoryPointer(mem, 0)
        #     im = cp.ndarray(shape=im.shape, memptr=mem_ptr, dtype=im.dtype)

        N, src_h, src_w, C = im.shape
        assert C == 3 # resize kernel only accept 3 channel tensors.
        dst_h, dst_w = self.imgsz

        if len(self.imgsz)!=2:
            raise "cuda resize target shape must be (h,w)"

        resize_scale = 1
        left_pad = 0
        top_pad = 0
        if self.use_padding:
            padded_batch = cp.zeros((N, dst_h, dst_w, C), dtype=self.im_dtype)
            if src_h / src_w > dst_h / dst_w:
                resize_scale = dst_h / src_h
                ker_h = dst_h
                ker_w = int(src_w * resize_scale)
                left_pad = int((dst_w - ker_w) / 2)
            else:
                resize_scale = dst_w / src_w
                ker_h = int(src_h * resize_scale)
                ker_w = dst_w
                top_pad = int((dst_h - ker_h) / 2)
        else:
            ker_h = dst_h
            ker_w = dst_w

        shape = (N, ker_h, ker_w, C)
        out = cp.empty(tuple(shape), dtype=cp.uint8)
        # define kernel configs
        block = (1024, )
        grid  = (ker_h, N)
        with cp.cuda.stream.Stream() as stream:
            cuResizeKer(grid, block,
                    (im, out,
                    cp.int32(src_h), cp.int32(src_w),
                    cp.int32(ker_h), cp.int32(ker_w),
                    cp.float32(src_h/ker_h), cp.float32(src_w/ker_w)
                    )
                )

            out = out.astype(self.im_dtype)
            if self.use_padding:
                if src_h / src_w > dst_h / dst_w:
                    padded_batch[:, :, left_pad:left_pad + out.shape[2], :] = out
                else:
                    padded_batch[:, top_pad:top_pad + out.shape[1], :, :] = out
            else:
                padded_batch = out

            # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            padded_batch = padded_batch[..., ::-1].transpose((0, 3, 1, 2))
            padded_batch = cp.ascontiguousarray(padded_batch)
            padded_batch /= 255

            stream.synchronize()

        return padded_batch


    def preprocess_cpu(self, im: np.ndarray):
        """
        Code here for reference, but no need to use the CPU here
        """
        im = np.stack([self.__resize(x) for x in im])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = cp.asarray(im)
        im = im.astype(cp.float16) if self.fp16 else im.astype(cp.float32)  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    

    def preprocess_yolo(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack([self.__resize(x) for x in im])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img


    def __resize(self, img):
        """
        Resize with padding in CPU
        """
        shape = img.shape[:2]
        new_shape = self.imgsz
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border


    def warmup(self):
        """
        Warm up the model by running one forward pass with a dummy input.
        """
        im = cp.empty((self.batch_size, 3, *self.imgsz), dtype=cp.float16 if self.fp16 else cp.float32)
        # im = torch.empty((self.batch_size, 3, *self.imgsz), dtype=torch.half if self.fp16 else torch.float, device="cuda:0")
        self.forward(im)


    def postprocess_detection(self, preds, imgsz, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds,
                                        self.conf,
                                        self.iou,
                                        agnostic=self.agnostic_nms,
                                        max_det=self.max_det,
                                        classes=self.classes)
        
        results = []
        for i, pred in enumerate(preds):
            # if not isinstance(orig_imgs, torch.Tensor):
            pred[:, :4] = scale_boxes(imgsz, pred[:, :4], orig_imgs[i].shape)
            save = orig_imgs[i].copy()
            for box in pred:
                p1 = box[:2].cpu().numpy().astype(np.int32)
                p2 = box[2:4].cpu().numpy().astype(np.int32)
                save = cv2.rectangle(save, p1, p2, (255, 0, 0), 1)
            cv2.imwrite(f"post{i}.png", save)
        return preds
    
    # def postprocess_yolo(self, preds, imgsz, orig_imgs):
    #     # [8, 84, 5880], [8, 3, 640, 448], list 8 (1080, 810, 3)
    #     """Post-processes predictions and returns a list of Results objects."""
    #     preds = ops.non_max_suppression(preds,
    #                                     self.conf,
    #                                     self.iou,
    #                                     agnostic=self.agnostic_nms,
    #                                     max_det=self.max_det,
    #                                     classes=self.classes)

    #     results = []
    #     for i, pred in enumerate(preds):
    #         orig_img = orig_imgs[i]
    #         pred[:, :4] = ops.scale_boxes(imgsz, pred[:, :4], orig_img.shape)
            # results.append(Results(orig_img=orig_img, path="img_path", names=["A"]*40, boxes=pred))
        # for i in range(len(results)):
        #     plot_args = {
        #         'line_width': 2,
        #         'boxes': True,
        #         'conf':True,
        #         'labels': True,
        #         'im_gpu': orig_imgs[i]
        #     }
        #     cv2.imwrite(f"post{i}.png", results[i].plot(**plot_args))
        # return results



    # print(pred[0].shape)
    # print(pred[1].shape)