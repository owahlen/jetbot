# JetBot

This fork of the original [JetBot repo](https://github.com/NVIDIA-AI-IOT/jetbot)
provides instructions on how to set up JetBot on a blank SD card.
It is based on information from the [JetBot Wiki](https://github.com/NVIDIA-AI-IOT/jetbot/wiki)
and includes changes from the [waveshare jetbot](https://github.com/waveshare/jetbot) fork.

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
* Reboot and click through gnome-initial-setup without changing anything
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
