import json
import logging
from pathlib import Path
from datetime import datetime

import tensorrt as trt

LOGGER = logging.getLogger(__name__)

metadata = {
    'author': 'BlueTensor S.R.L.',
    'date': datetime.now().isoformat()
}

def main(args):
    f_onnx = args.file
    assert Path(f_onnx).exists(), f'ONNX file {f_onnx} not found'
    assert f_onnx.endswith('.onnx'), f'invalid ONNX file: {f_onnx}'
    f = Path(args.file).with_suffix('.engine')  # TensorRT engine file
    metadata["batch"] = args.batch
    metadata["imgsz"] = args.imgsz
    metadata["task"] = args.task

    logger = trt.Logger(trt.Logger.INFO)
    if args.verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(f_onnx):
        raise RuntimeError(f'failed to load ONNX file: {f_onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        LOGGER.info(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    LOGGER.info(
        f'building FP{16 if builder.platform_has_fast_fp16 and args.half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and args.half:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # trt.init_libnvinfer_plugins(logger, '')
    # PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

    # def get_trt_plugin(plugin_name):
    #     plugin = None
    #     for plugin_creator in PLUGIN_CREATORS:
    #         if plugin_creator.name == plugin_name:
    #             lrelu_slope_field = trt.PluginField("neg_slope", np.array([0.1], dtype=np.float32), trt.PluginFieldType.FLOAT32)
    #             field_collection = trt.PluginFieldCollection([lrelu_slope_field])
    #             plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    #     return plugin

    # builder.max_workspace_size = 2**20
    # input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 1))

    # lrelu = network.add_plugin_v2(inputs=[input_layer], plugin=get_trt_plugin("LReLU_TRT"))
    # lrelu.get_output(0).name = "outputs"
    # network.mark_output(lrelu.get_output(0))

    # Write file
    with builder.build_serialized_network(network, config) as serialized_network, open(f, 'wb') as t:
        # Metadata
        meta = json.dumps(metadata)
        t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        t.write(meta.encode())
        # Model
        t.write(serialized_network)

    return f, None

if __name__ == "__main__":
    """
    Example usage:
    python tensorrt_export.py models/FastSAM.onnx --verbose --workspace 4 --half --batch 8 --imgsz 640 640 --task detect
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='model.onnx path')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--workspace', type=int, default=4, help='workspace size (GB)')
    parser.add_argument('--half', action='store_true', help='quantize to FP16')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640], help='image size')
    parser.add_argument('--task', type=str, default='detect', help='task type', choices=['detect', 'classify', 'segment'])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    main(args)