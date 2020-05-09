# JetBot

This fork of the original [JetBot repo](https://github.com/NVIDIA-AI-IOT/jetbot)
provides instructions on how to set up JetBot for JetPack 4.4 on a blank SD card.
It is based on information from the [JetBot Wiki](https://github.com/NVIDIA-AI-IOT/jetbot/wiki).

The repo includes patches from the [waveshare jetbot](https://github.com/waveshare/jetbot) 
fork to show battery information in the OLED display.
Since Jetpack 4.4 comes with TensorRT 7, it necessary to rebuild the TensorRT engine.

## 1. Write Image to microSD Card
The JetBot software will be installed on top of the
[Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write).
Download the [SD card image](https://developer.nvidia.com/jetson-nano-sd-card-image) and
write it onto the microSD card using [etcher](https://www.balena.io/etcher/).
Alternatively start the download with the following command:
```
wget https://developer.download.nvidia.com/embedded/L4T/r32_Release_v4.2/nv-jetson-nano-sd-card-image-r32.4.2.zip
```

## 2. Basic Setup of Jetson 
Plug the microSD card into the Jetson board in addition to keyboard, mouse, monitor and boot.
Read the instructions on
[Setup and First Boot](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup)
and
[Create-SD-Card-Image-From-Scratch](https://github.com/NVIDIA-AI-IOT/jetbot/wiki/Create-SD-Card-Image-From-Scratch)
on the Jetson wiki page.

In summary:
* Accept NVIDIA Jetson software EULA
* Select system language, keyboard layout, WiFi, and time zone
* Use `jetbot` as username, password, and computer name also select the checkbox to login automatically
* Leave APP partition size on its default value
* Reboot and click through _gnome-initial-setup_ without changing anything
* Find out the IP address of the device with the command: `hostname -I | awk '{print $1}'`
* The jetbot should now be reachable via `ssh` from the host and all peripherals can be disconnected 

## 3. Setup of Software
Copy the setup script from the host to the jetbot and login onto the board:
```
scp scripts/create-sdcard-image-from-scratch.sh jetbot@<ip address of jetbot>:
ssh jetbot@<ip address of jetbot>
```

Upgrade the system and run the setup script:
```
sudo apt-get update
sudo apt-get upgrade -y
sudo bash create-sdcard-image-from-scratch.sh
```

## 4. Configure Power Mode
On the jetbot select 5W power mode
```
sudo nvpmodel -m1
```
Verify the correct power mode setting
```
sudo nvpmodel -q
```

## 5. Setup Object Detection
Nvidia proposes to use [TensorRT](https://developer.nvidia.com/tensorrt)
in order to achieve high performant object detection on the Jetson board.
A high level overview of of the required looks like this:

1. Provide a frozen Tensorflow model (_pb format_).
   This can be achieved in several ways:
    1. Create, train, and freeze a custom developed model 
    2. Download a pre-trained model e.g. from the
       [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)       
2. Convert the frozen Tensorflow model into a 
   [UFF model](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_tf_python),
   an intermediate file format defined by TensorRT.
3. From the UFF model build and serialize an 
   [engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_python)
   that is optimized for running on the target platform.
   Note that serialized engines are not portable across platforms or TensorRT versions.
4. Load and execute the engine for inference on the target platform.

This repository contains wrapper classes that load and execute the engine.

### 5.1. Create the detector engine
In order to create the engine the 
[AastaNV/TRT_object_detection](https://github.com/AastaNV/TRT_object_detection)
repository is used.
```
cd ~
git clone https://github.com/AastaNV/TRT_object_detection
cd TRT_object_detection
```
All dependencies except for `pycuda` are already installed in versions required by Tensorflow.
```
sudo pip3 install pycuda
```
To build the UFF model the frozen graph is required in the directory `model`.
```
mkdir model
cd model
curl -s http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz | tar xvz
cd ..
```
TensorRT 7 requires to patch graphsurgen in a different way than explained in the repo.
Add the `patched` line to `/usr/lib/python3.6/dist-packages/graphsurgeon`:
```
diff --git a/node_manipulation.py b/node_manipulation.py.orig
index 7fea2f3..4f74842 100644
--- a/node_manipulation.py
+++ b/node_manipulation.py.orig
@@ -39,7 +39,6 @@ def update_node(node, name=None, op=None, trt_plugin=False, **kwargs):
     '''
     node.name = name or node.name
     node.op = op or node.op or node.name
-    node.attr["dtype"].type = 1 # patched
     for key, val in kwargs.items():
         if isinstance(val, tf.DType):
             node.attr[key].type = val.as_datatype_enum
```
For the conversion to UFF file, TensorRT requires a plugin called `FlattenConcat`.
There is a shared library `lib/libflattenconcat.so` folder that provides the plugin.
Unfortunately it has been built with TensorRT 5 and is not compatible with TensorRT 7.
We have to copy the plugin that has been built in the jetbot repository into this location.
```
cp ~/jetbot/build/lib/jetbot/ssd_tensorrt/libssd_tensorrt.so \
   ~/TRT_object_detection/lib/libflattenconcat.so
```

Now run `main.py` in the toplevel of the repository.
```
python3 main.py /usr/src/tensorrt/samples/python/uff_ssd/images/image2.jpg
```
The script runs several minutes and tries to open a window showing results.
Most important however it produces the engine `TRT_ssd_mobilenet_v2_coco_2018_03_29.bin`.
Copy this file as engine into the `object_following` notebook:
```
cp ~/TRT_object_detection/TRT_ssd_mobilenet_v2_coco_2018_03_29.bin \
   ~/jetbot/notebooks/ssd_mobilenet_v2_coco.engine
```

