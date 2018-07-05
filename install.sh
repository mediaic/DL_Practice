##########################################################
# ****** This script is ONLY for Ubuntu 16.04 LTS ****** #
##########################################################

############ Cuda setup ############
# Download specified CUDA driver package
CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo wget -O ${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG} 
# Register driver package on your machine
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-get update
# install CUDA driver
sudo apt-get install cuda-drivers
sudo apt-get install cuda=9.0.176-1
# Register & install CuDNN library
wget 'https://www.dropbox.com/s/bl3uaaj6az1v8gs/libcudnn7_7.0.5.15-1%2Bcuda9.0_amd64.deb'
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo apt install nvidia-cuda-toolkit

############ Install deep learning frameworks ############
# upgrade pip
sudo pip3 install --upgrade pip
# install tensorflow 1.6
sudo pip3 install tensorflow-gpu==1.6
# install pytorch 0.4.0
sudo pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
sudo pip3 install torchvision
