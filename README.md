# GoProMoCap_v2

##Dependencies

1.OpenCV <br /> 
2.numpy <br /> 
3.Scipy - from least_square optimisation during bundle adjustment <br /> 
4. Pandas


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
