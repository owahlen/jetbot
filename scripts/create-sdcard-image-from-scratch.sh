#!/bin/bash

password='jetbot'

# Keep updating the existing sudo time stamp
sudo -v
while true; do sudo -n true; sleep 120; kill -0 "$$" || exit; done 2>/dev/null &

# Enable i2c permissions
sudo usermod -aG i2c $USER

# Install pip and some python dependencies
sudo apt-get update
sudo apt install -y python3-pip python3-pil
sudo pip3 install --upgrade numpy

# Install the pre-built TensorFlow pip wheel
sudo apt-get update
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install -U pip testresources setuptools
sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
#sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow

# Install the pre-built PyTorch pip wheel
sudo apt-get install -y libopenblas-base libopenmpi-dev
wget https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl -O torch-1.5.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install Cython
sudo pip3 install torch-1.5.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.5.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision package
git clone https://github.com/pytorch/vision
cd vision
git checkout v0.6.0
sudo python3 setup.py install
cd
rm -rf vision

# Install traitlets (master, to support the unlink() method)
sudo python3 -m pip install git+https://github.com/ipython/traitlets@master

# Install jupyter lab
sudo apt-get install -y curl
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo pip3 install jupyter jupyterlab
sudo jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab --generate-config
#jupyter notebook password
python3 -c "from notebook.auth.security import set_password; set_password('$password', '$HOME/.jupyter/jupyter_notebook_config.json')"

# install jetbot python module
cd
sudo apt install -y python3-smbus
git clone https://github.com/owahlen/jetbot.git
cd ~/jetbot
sudo apt-get install -y cmake
sudo python3 setup.py install 

# Install jetbot services
cd jetbot/utils
python3 create_stats_service.py
sudo mv jetbot_stats.service /etc/systemd/system/jetbot_stats.service
sudo systemctl enable jetbot_stats
sudo systemctl start jetbot_stats
python3 create_jupyter_service.py
sudo mv jetbot_jupyter.service /etc/systemd/system/jetbot_jupyter.service
sudo systemctl enable jetbot_jupyter
sudo systemctl start jetbot_jupyter

# Make swapfile
cd 
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'

# Copy JetBot notebooks to home directory
cp -r ~/jetbot/notebooks ~/Notebooks

# Install fan-daemon if you have a PWM fan connected to the Jetson
cd
git clone https://github.com/kooscode/fan-daemon.git
cd fan-daemon
make
sudo bash -c './install.sh'
