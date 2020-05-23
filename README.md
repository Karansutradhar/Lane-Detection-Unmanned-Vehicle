ENPM673: Perception for Autonomous Robotics Project 2

#### Instructions to run the program for Project 2
file name : project2_problem1.py
file name : project2_problem2.py


Running instructions for problem 1:

1) To execute the code for problem 1, run the command `python3 project2_problem1.py`.
You will get the output which is being compared with the original video.
2) Make sure the .py file and the video are in the same directory.
3) The code will also write the output into the current directory.
4) The program will destroy all the windows once the video has finished playing.

Running instructions for problem 2:

1) To run the code for problem two, provide the relative path to the input video as an argument.
For example, if the video is a file named `challenge_video.mp4` in a directory called Videos
that is in the same directory as the code, then you have to type the following command:

    `python3 project2_problem2.py Videos/challenge_video.mp4`

in the terminal.
2) Make sure the py file and the directory containting the videos are in the same directory.
3) Additionally you can also type "-h" for help:
    
    `python3 project2_problem2.py -h`

4) We have converted the frames that were provided to us into a video and renamed it as DayDrive.mp4
5) After processing, OpenCV will display the original video alongside the final result.
OpenCV will loop the video indefinitely until you decide to stop the program.
Use Control-C in the terminal to exit the video.
6) The code will write the output into the current directory before processing the video.

We imported the following libraries:

1) numpy 
2) cv2
3) sys
4) math
5) argparse

Results for problem 1:
You will be able to see a pop up with the brighter video along with the original video 

Results for problem 2:
Based on your choice of video you will see a pop up with detected lanes and turn prediction being compared with the original video. 


The video output of our program can be found in the links below:
 * Day drive result:  https://drive.google.com/open?id=1rCX3lqPK5dk8fWKKc79EiC4bV_fdv5ed
 * Challenge result:  https://drive.google.com/open?id=10ieWuzF8WZSwEkLLht0K0qhOLmLJqB8c
