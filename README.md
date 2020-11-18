# GoProMoCap_v2

##Dependencies

1.OpenCV <br /> 
2.numpy <br /> 
3.Scipy - from least_square optimisation during bundle adjustment <br /> 
4. Pandas

#User Inputs
Place videos in a their own filepath in one folder called SourceVideos. 
If videos are multiple parts keep in same folder with '..part1' ,'..part2', etc as video names.
Open config.py and change dataFile variable to the path of your videos. 
Change DLCConfigPath to the path of your trained Deeplabcut network and openPoseFolderPath to the path where you have openpose downloaded to. 
Enter the amount of cameras you recorded with, if there is any rotation in the way the cameras recorded and whether or not you want to track face and hands in openpose.
 
#ProcessVideos.py 
After setting the config variables correctly, open main.py and run script. Multiple functions from ProcessVideos.py are called

concatVideos: If videos were recorded in multiple parts, the function concatenates each part into a single video

trimVideos: Function finds the brightest frame from the first and last third of each video and trims the videos to those two frames
    trimVideos function only necessary if syncing cameras through flash bulb
    
runDeepLabCut: Runs videos through the specified Deeplabcut network, saves an npy file for each camera that has pixel coords of each point trcaked in DLC 

runOpenPose: Run video through openpose, saves raw openpose data with a separate json file for each frame

ParseOpenPose: Parses through each json file, saves every frame from each camera to an HDF5 file and also a separate npy file for each camera.


Create a data --> OpenPose folder. Place the downloaded npy files(OP_CamE/F/G/H) from https://drive.google.com/drive/u/1/folders/1N7qq-WiGjAzeLoODp9zlgSXMnFsMt-RZ and rename them (OP_CamE.npy --> OP_Cam1.npy,OP_CamF.npy --> OP_Cam2.npy,OP_CamG.npy --> OP_Cam3.npy, OP_CamH.npy --> OP_Cam4.npy........) in the OpenPose folder.
<br /> 
<br /> 
<br /> 
Run python main.py to get the skeleton motion capture data and visulization of the skeleton.<br /> 
<br /> 
<br /> 
<br /> 
<br /> 
if need visulization of skeleton together with the original video, Create data --> SourceVideos folder(the same data folder as above). Download videos from https://drive.google.com/drive/u/1/folders/1g9joFk_gDh8VWsCzZDHVpEN1-I9-tF75. (over 80GB)  rename them (OpenPoseCamE.avi --> Cam1.avi,OpenPoseCamF.avi --> Cam2.avi.......).
replace line 16 in the main file 'from visualize_without_video import Vis' with 'from visualize import Vis' to get the visulization of the skeleton with oringinal videos.
<br /> 
