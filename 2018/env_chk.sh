# Status
echo -n 'Python Version '
python3 --version | cut -d" " -f2
cat /usr/local/cuda/version.txt
echo -n 'Tensorflow Version '
pip3 freeze | grep tensorflow-gpu | cut -d"=" -f3
echo -n 'Pytorch Version '
pip3 freeze | grep 'torch==' | cut -d"=" -f3