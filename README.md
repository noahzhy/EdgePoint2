# EdgePoint2: Compact Descriptors for Superior Efficiency and Accuracy

EdgePoint2 is a series of lightweight keypoint detection and description neural networks specifically tailored for edge computing applications. We provide 14 sub-models with various network configurations (Tiny/Small/Medium/Large/Enormous) and compact descriptor dimensions (32/48/64) to accommodate diverse usage requirements.

## Install Dependencies

```shell
pip install -r requirements.txt
```

## Usage

To run the demo using a camera, use the following command:

```shell
python demo_seq.py camera 
```

Alternatively, to run with a video file, use:

```shell
python demo_seq.py PATH_TO_VIDEO_FILE
```

You can add ```--model``` to select the sub-model from {T32, T48, S32, S48, S64, M32, M48, M64, L32, L48, L64, E32, E48, E64}. The letter indicates the model size, while the number denotes the output descriptor dimension.

Export the model to ONNX format using the following command:

```shell
python export.py --model T32 --input_height 640 --input_width 480 --topK 2048 --output_dir ./onnx
```
