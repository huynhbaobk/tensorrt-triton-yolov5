# PSENet

**preprocessing + inference + postprocessing = 30ms** with fp32 on Tesla P40. 
The original Tensorflow implementation is [tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet). A TensorRT Python api implementation is [TensorRT-Python-PSENet](https://github.com/upczww/TensorRT-Python-PSENet).

## Key Features
- Generating `.wts` from `Tensorflow`.
- Dynamic batch and dynamic shape input.
- Object-Oriented Programming.
- Practice with C++ 11.


<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/105487078-821d6800-5cea-11eb-87dc-e3317a941763.jpeg">
</p>

## How to Run

* 1. generate .wts

  Download pretrained model from https://github.com/liuheng92/tensorflow_PSENet
  and put `model.ckpt.*` to `model` dir. Add a file `model/checkpoint` with content
    ```
    model_checkpoint_path: "model.ckpt"
    all_model_checkpoint_paths: "model.ckpt"
    ```
    Then run

    ```
    python gen_tf_wts.py
    ```
    which will gengerate a `psenet.wts`.
* 2. cmake and make

  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
* 3. build engine and run detection
  ```
  cp ../psenet.wts ./
  cp ../test.jpg ./
  ./psenet -s  // serialize model to plan file
  ./psenet -d  // deserialize plan file and run inference
  ```

## Known Issues
None

## Todo

* use `ExponentialMovingAverage` weight.
