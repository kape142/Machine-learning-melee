# Machine-learning-melee
An attempt at making an AI for Super Smash Bros. Melee using Q-learning

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

3) Legally acquire an ISO picture of Melee v1.02 NTSC.

4) Clone this repository

5) Run q_learning.py to start the learning!

In order to test the trained agent against a random agent, make a copy of your Q table with a different name, and input 
that name into the "stored_filename" variable in q_table_benchmarking.py, and start the script.

In order to graph the data generated under the learning, set the parameter "stored_filename" in the call to show_all() 
to the name of the "stored_filename" variable used in the learning process, and start the script

