# Machine-learning-melee
An attempt at making an AI for Super Smash Bros. Melee using machine learning


# Installing Libmelee on Linux ubuntu 18.04
1) First of all install CMake:

Download cmake-3.13.0-rc1.tar.gz from: https://cmake.org/download/
Run the following commands in terminal:
cd \folder\which\contains\the\.tar.gz\filename
tar xvzf PACKAGENAME.tar.gz
./configure
make
sudo make install

2) Install dolphin:
Run the following commands in terminal:
git clone --single-branch -b memorywatcher https://github.com/altf4/dolphin.git
cd \path\to\dolphin\folder\just\cloned

mkdir build
cd build
cmake ..
make
sudo make install

3) Download iso picture of Melee v1.02 NTSC.
4) Install via pip: (or by cloning this repo for the very latest) pip3 install melee

5) Run example.py to verify that the installation was successful. Done with typing Python3 example.py in terminal.
