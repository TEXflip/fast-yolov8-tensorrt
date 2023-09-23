import time

import cv2
import cupy as cp
import numpy as np
from yolov8.yolov8 import YOLOv8trt

np.bool = bool

if __name__ == "__main__":
    MODEL_PATH = "./yolov8/models/yolov8n_b8_s640-448.engine"
    # MODEL_PATH = "../models/yolov8n.engine"
    N_ITERATIONS = 10

    model = YOLOv8trt(MODEL_PATH)
    BATCH_SIZE = model.batch_size

    image = cv2.imread('./test/test_data/image.jpg')
    # image_batch = np.array([image] * BATCH_SIZE)
    image_batch = cp.array([image] * BATCH_SIZE)


    model.warmup()

    start_time = time.time()
    for _ in range(N_ITERATIONS):
        pred = model.predict(image_batch)
    tot_time = (time.time() - start_time) / (N_ITERATIONS * BATCH_SIZE)

    for prof in model.profilers:
        prof.t /= N_ITERATIONS * BATCH_SIZE

    print('')
    print('')
    print(f"preprocess time per image: {round(model.profilers[0].t * 1000, 2)} ms ({round(1/model.profilers[0].t, 2)} fps)")
    print(f"inference time per image: {round(model.profilers[1].t * 1000, 2)} ms ({round(1/model.profilers[1].t, 2)} fps)")
    print(f"postprocess time per image: {round(model.profilers[2].t * 1000, 2)} ms ({round(1/model.profilers[2].t, 2)} fps)")
    print(f"tot time per image: {round(tot_time * 1000, 2)} ms ({round(1/tot_time, 2)} fps)")