# YOLOv5 on Triton Inference Server with TensorRT

This repository shows how to deploy YOLOv5 as an optimized [TensorRT](https://github.com/NVIDIA/tensorrt) engine to [Triton Inference Server](https://github.com/NVIDIA/triton-inference-server).

This project based on [isarsoft
 yolov4-triton-tensorrt](https://github.com/isarsoft/yolov4-triton-tensorrt) and [Wang Xinyu - TensorRTx](https://github.com/wang-xinyu/tensorrtx)

## Build TensorRT engine

Run the following to get a running TensorRT container with our repo code:

```bash
cd tensorrt-triton-yolov5
bash launch_tensorrt.sh
```

### Or build docker from Dockerfile
```bash
cd tensorrt-triton-yolov5
sudo docker build -t baohuynhbk/tensorrt-20.08-py3-opencv4:latest -f tensorrt.Dockerfile .
```

Docker will download the TensorRT container. You need to select the version (in this case 20.08) according to the version of Triton that you want to use later to ensure the TensorRT versions match. Matching NGC version tags use the same TensorRT version.

Inside the container the following will run:
```bash
bash convert.sh
```
This will generate a file called `yolov5.engine`, which is our serialized TensorRT engine. Together with `libmyplugins.so` we can now deploy to Triton Inference Server.

## Deploy to Triton Inference Server

### Start Triton Server

Open an terminal

```bash
bash run_triton.sh
```

### Client
Should install tritonclient first:
```bash
sudo apt update
sudo apt install libb64-dev

pip install nvidia-pyindex
pip install tritonclient[all]
```
Open another terminal.
This repo contains a python client.
```bash
cd triton-deploy/clients/python
python client.py -o data/dog_result.jpg image data/dog.jpg
```

### Benchmark

To benchmark the performance of the model, we can run [Tritons Performance Client](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/optimization.html#perf-client).

To run the perf_client, install the Triton Python SDK (tritonclient), which ships with perf_client as a preinstalled binary.

```bash
# Example
perf_client -m yolov5 -u 127.0.0.1:8001 -i grpc --shared-memory system --concurrency-range 32
```

The following benchmarks were taken on a system with `NVIDIA 2070 Ti` GPU.
Concurrency is the number of concurrent clients invoking inference on the Triton server via grpc.
Results are total frames per second (FPS) of all clients combined and average latency in milliseconds for every single respective client.
