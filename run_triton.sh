mkdir -p triton-deploy/models/yolov5/1/
mkdir -p triton-deploy/plugins

cp tensorrtx/yolov5/build/yolov5m6.engine triton-deploy/models/yolov5/1/model.plan
cp tensorrtx/yolov5/build/libmyplugins.so triton-deploy/plugins/

sudo docker run --gpus "device=0" --name tritonserver-20.08 --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p8220:8000 -p8221:8001 -p8222:8002 -v$(pwd)/triton-deploy/models:/models -v$(pwd)/triton-deploy/plugins:/plugins --env LD_PRELOAD=/plugins/libmyplugins.so nvcr.io/nvidia/tritonserver:20.08-py3 tritonserver --model-repository=/models --strict-model-config=false --grpc-infer-allocation-pool-size=16
