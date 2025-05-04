import os, glob, sys, random
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# set seed
np.random.seed(0)
random.seed(0)


def inference_onnx_model(model_path, img_path, target_size=(480, 640)):
    # Load image
    img = Image.open(img_path).convert('RGB')
    # resized via width and keep aspect ratio
    t_w, t_h = target_size
    w, h = img.size
    scale = min(t_w / w, t_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Create a new image with the target size and paste the resized image onto the center
    new_img = Image.new('RGB', (t_w, t_h), (0, 0, 0)) # Use black background
    left = (t_w - new_w) // 2
    top = (t_h - new_h) // 2
    new_img.paste(img, (left, top))
    img = new_img

    resize_img = np.array(img)
    # Prepare input: convert to NCHW format
    img_input = np.transpose(resize_img, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    # Run inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_input})

    return output, resize_img


def test_onnx_model_speed(model_path, inputs, warm_up=20, test=200, force_cpu=True):
    # Set ONNX Runtime options for better performance
    options = ort.SessionOptions()
    # options.enable_profiling = True
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create session with optimized options
    provider_options = ['CPUExecutionProvider'] if force_cpu else ort.get_available_providers()

    session = ort.InferenceSession(model_path, options, providers=provider_options)
    # run one inference to get the output shape
    session.run(None, inputs)

    # Combined warm-up and test phase
    times = []
    total_iterations = warm_up + test
    for i in range(total_iterations):

        if i < warm_up:
            session.run(None, inputs)
            continue

        start = perf_counter()
        session.run(None, inputs)
        end = perf_counter()
        times.append(end - start)

    times = np.array(times) # Convert to numpy array for subsequent calculations

    # Calculate statistics
    # sort the times and remove the first 10% and last 10% of the times
    times = np.sort(times)[int(test * 0.3):int(test * 0.7)]
    avg_time = np.mean(times)
    print(f'Avg time: {avg_time * 1000:.2f} ms')
    print(f'Min time: {min(times)*1000:.2f} ms')
    print(f'Max time: {max(times)*1000:.2f} ms')


def draw_points(img, points, size=2, color=(255, 0, 0), thickness=-1):
    for p in points:
        cv2.circle(img, tuple((int(p[0]), int(p[1]))), size, color, thickness)
    return img


if __name__ == '__main__':
    model_path = 'onnx/ep2_T32.onnx'
    # model_path = 'onnx/xfeat_2048_640x480.onnx'
    input_shapes = {
        'images': np.random.random((1, 3, 640, 480)).astype(np.float32)
        # 'images': np.random.random((1, 3, 1280, 960)).astype(np.float32)
    }

    # model_path = 'onnx/lighterglue_L3.onnx'
    # n_kpts = 1024
    # kpt0 = np.random.random((1, n_kpts, 2)).astype(np.float32)
    # desc0 = np.random.random((1, n_kpts, 64)).astype(np.float32)
    # kpt1 = np.random.permutation(kpt0)
    # desc1 = np.random.permutation(desc0)
    # input_shapes = {
    #     'kpts0': kpt0,
    #     'kpts1': kpt0,
    #     'desc0': desc0,
    #     'desc1': desc0
    # }

    test_onnx_model_speed(model_path, input_shapes)

    img_path = random.choice(glob.glob('assets/002.*g'))
    output, resize_img = inference_onnx_model(model_path, img_path, target_size=(480, 640))

    outputs = {
        'keypoints': output[0],
        'scores': output[1 if len(output[1].shape) == 1 else 2],
        'descriptors': output[2 if len(output[1].shape) == 1 else 1]
    }

    scores = outputs["scores"]
    kpts = outputs["keypoints"]

    # kpts = output[0][(scores > 0.5) & (scores < 1.0)]
    kpts = kpts[(scores > 0.1) & (scores < 1.0)]

    print(f'scores: {scores}')
    print(f'keypoints shape: {kpts.shape}')

    # to visualize keypoints
    img = draw_points(resize_img, kpts)
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("test.jpg", img)
