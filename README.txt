download trainset/ at 
download testset/ at 


FILES AND FOLDERS AND ASSUMPTIONS
====================================
orientation.py  : Main python script
                  - Assumes "testset" folder exists in the same folder as this file.
                        - Change this source folder in LINE 22 of code if you want to
                          load data from a different source
                  - Assumes source folder contains "imu" subfolder
myquaternion.py : Contains quaternion function implemtations



EXECUTION INSTRUCTIONS
========================
Make sure all requirements above are met, then run "python orientation.py" from terminal

WHAT SCRIPT DOES AND OUTPUTS
-----------------------------
- Reads IMU data (.p files) from "testset/imu" folder. 
- Reads cam data (.p files) from "testset/cam" folder IF IT EXISTS. 
- Reads vicon data (.p files) from "testset/vicon" folder IF IT EXISTS. 
- For each data set 
     - Launches one window with the following four images
          - Roll Plot
          - Pitch Plot
          - Yaw Plot
          - Panorama image
     - Prints to terminal
          - Step being performed and it's status
     
- CLOSE DISPLAY IMAGE FOR EXECUTION TO CONTINUE TO NEXT DATASET
         