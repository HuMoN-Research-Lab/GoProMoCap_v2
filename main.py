import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from ops import get_RT_mtx,video_loader
from config import video_resolution
import glob
import pickle 
import cv2.aruco as aruco
import os
import pickle
from ops import toCsv,vec2skewMat,inverseH,R_t2H,get_RT_mtx,video_loader,get_TransMat,triangulate,aruco_detect,charuco_detect,SBA,SBA2
from config import base_Cam_Index,num_of_cameras,video_resolution,Len_of_frame,SAVE_FOLDER,start_frame,SourceVideoFolder,Pixel_coord_FIlE_PATH,Source_video_List, rotation, dataFile
#from visulize_with_out_head import Vis
#from visualize_without_video import Vis
from scipy.optimize import least_squares
import time
from scipy.sparse import lil_matrix
from ProcessVideos import concatVideos, trimVideos, runDeepLabCut, runOpenPose, Parse_Openpose, create_calibration


class Exceptions(Exception):
    pass

#==================load image from videos to calibration folder,these image will be used to calibrate camera/pose estimation
#video_loader(SourceVideoFolder)

def VIS(coords):
    ax1 = plt.subplot(1,1,1,projection='3d') #3D

    temp = ax1.scatter3D(coords[0,:,0], coords[0,:,1], coords[0,:,2], marker='o', cmap='hot',s=200)
    plt.show()

def VIS_temp(coords):
    ax1 = plt.subplot(1,1,1,projection='3d') #3D

    temp = ax1.scatter3D(coords[:,0], coords[:,1], coords[:,2], marker='o', cmap='hot',s=200)
    plt.show()






def calibration(calibration_folder_path):
     
    def Charuco_Corners_filter(allIds,allCorners):
        """
        allIds: list contains ids of each camera
        allCorners: list contains corners of each camera
        input should be corners and ids of all cameras. pick the ids with shortest length and then recollect the corners.

        return
        ret: filtered corners
        val: length of valid corners
        """

        def dict_builder(ids, corners):
            """
            build a dictionary for each camera {id:coord,...}
            """
            
            if len(ids) != len(corners):
                raise Exceptions('len of ids must match len of corners')
        
            ids = ids.reshape((-1,))
            ret = {}
            ind = 0 
            for id in ids:
                ret[id] = corners[ind]
                ind += 1

            return ret
        
        if len(allIds) != len(allCorners):
            raise Exceptions('len of ids must match len of corners')


        L = [] #store length of each camera ids
        reshaped_Corners = []#store reshaped corners (length,2)
        dicts = []#store dictionaries {id:corners}

        for id in allIds:
            #get length of each list of ids
            l = len(id)
            #store length of each list of ids in a new list
            L.append(l)
        
        for i in range(len(allCorners)):
            #length of ith corner
            length = L[i]
            #reshape corners as (length,2)
            corner_coord = allCorners[i].reshape((length,2))
            reshaped_Corners.append(corner_coord)

        assert len(L) == len(reshaped_Corners)

        
        #build dictionary for each pair of ids/corners
        for i in range(len(L)):
            d = dict_builder(allIds[i],reshaped_Corners[i])
            dicts.append(d)

        #return the shortest length and its index(camera indx), val:shortest length/idx: index
        val, idx = min((val, idx) for (idx, val) in enumerate(L))

        tempID = allIds
        targetIDs = []
        #24:the charucoboard we currently using has ID 0-23
        for i in range(24):
            boo = True
            for indx in range(len(tempID)):
                c = 0
                while tempID[indx][c] != i:
                    c+=1
                    if c == len(tempID[indx]):
                        boo = False
                        break
                else:
                    np.delete(tempID[indx],c)
                
            if boo:
                targetIDs.append(i)


        #print(idx,':',val)
        
        #minmum set of ids which is valid in all camera view
        target_ids = allIds[idx].reshape((-1,))

        target_ids = targetIDs

        ret = []#list stores filtered corners, corners have shape(val,2)
        
        #===========for debug
        val = len(target_ids)
        
        
        #extract each targetId's corresponding corners for every camera view
        for dict in dicts:
            #each camera view
            temp = np.zeros((val,2))#temp numpy array stores filtered 
            count = 0
            #each target id
            for id in target_ids:
                temp[count] = dict[int(id)].reshape((2))
                count+=1
            temp = temp.reshape((1,val,2))
            ret.append(temp)
            
            
        return ret,val
    
    
    def get_cam_info(path):
        """
        input
        path: calibration fold that contains image(s) of charucoboard


        return
        Ids: charucoboard ids, list of integers
        corners: charucoboard corners, contains pixel coordinates of each corners
        H_mat:(4x4)homogenies transformation matrixs
        K: (3x3) camera matrix
        """

        K,dist,rvecs,tvecs,corners,Ids = charuco_detect(path,video_resolution)
        corners = np.array(corners)
        RoMat,_ = cv2.Rodrigues(rvecs) #convert (3,1) rvec to (3,3) rotation matrix
        H_mat = R_t2H(RoMat,tvecs) #this funciton in ops line 87, combine rotation matrix and tvec into a Homogeniens transformation matrix


        return Ids,corners,H_mat,K

    def bundle_adjustment(frames,reference_pixelCoords,input,calibration_points):
        """
        input____
        frames: integer,number of frames to be optimized
        reference_pixCoords: 3 dimensional numpy array (frames,KeyPoints,xyz), pixel coordinates provided by openpose/dlc
        input: 1d array, falllented 3d pixelCoords and projection matrix

        output___
        optimized 3d points
        optimized projection matrix
        """

        def fun(input):
            _3dpoints = input[:frames*3*calibration_points].reshape((-1,calibration_points,3))#reshape back to(len,points,3)
            projectionMat = input[frames*calibration_points*3:]
            
            #attach 1 to the end of xyz
            temp = np.ones((_3dpoints.shape[0],_3dpoints.shape[1],1))
            x = np.concatenate((_3dpoints,temp),axis=2)
            shape = np.squeeze(reference_pixelCoords).shape
            #reprojection = np.zeros(shape)

            l = len(projectionMat)//num_of_cameras
            ProjMat = []#store reshaped projection matrix

            for i in range(num_of_cameras):
                j = i+1
                P = projectionMat[i*l:j*l].reshape((3,4))
                ProjMat.append(P)
            
            ProjMat = np.array(ProjMat)
            reprojection = []

            for i in range(len(ProjMat)):
                temp = x.dot(ProjMat[i].T)
                reprojection.append(temp)

            reprojection = np.stack(reprojection,axis = 0)
            reprojection = reprojection[:,:,:,:2] / reprojection[:,:,:,2,np.newaxis]
            
            res = (reference_pixelCoords-reprojection)

            return res.ravel()
        
        residual = fun(input)
        plt.plot(residual)
        plt.show()

        t0 = time.time()
        res = least_squares(fun,input,verbose=2, x_scale='jac', ftol=1e-6, method='trf')
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))
        plt.plot(res.fun)
        plt.show()
        
        param = res.x
        print(param.shape)
        optimized_3D = param[:frames*3*calibration_points]
        optimized_ProjMat = param[frames*3*calibration_points:]
        coords = optimized_3D.reshape((-1,calibration_points,3))


        return coords,optimized_ProjMat



    temp_path = [x[0] for x in os.walk(calibration_folder_path)]
    allPath = []
    for p in temp_path:
        if not p.startswith('.'):
            allPath.append(p)
    allPath = allPath[1:]
    allPath.sort(key=lambda f: int(re.sub('\D', '', f)))#sort path
    cam_number = len(allPath)

    if cam_number != num_of_cameras:
        raise Exceptions('number of calibration folder must equal to number of cameras')
    

    allIds = []#stores list of charucoboard Ids of each camera view
    allCorners = []#stores list of charucoboard corners of each camera view
    Hmats = []#stores each (4x4) homogenies transformation matrixs
    Kmats = []#stores each (3x3) camera matrix

    camera_positions = {}#export each cameras tvec and rvec

    #read each camera calibration folder
    for path in allPath:
        fullPath = path+'/*.jpg'
        Ids,corners,H_mat,K = get_cam_info(fullPath)
        camera_positions[path] = inverseH(H_mat)#first 3 coloumns are rotation matrix, the last column is tvec
        allIds.append(Ids)
        allCorners.append(corners)
        Hmats.append(H_mat)
        Kmats.append(K)
    
    assert len(allIds) == len(allCorners) == len(Hmats) == len(Kmats)
    filered_corners,calibration_points = Charuco_Corners_filter(allIds,allCorners)


    #stack projection matrix and pixel coordinates for triangulation
    transMats = get_TransMat(Hmats)
    ProjectionMats = []#stores each (camera Mat * transMat)
    for i in range(len(Kmats)):
        P = np.dot(Kmats[i],transMats[i])
        ProjectionMats.append(P)
    
    #project points format should be (frame,#_of_keypoints,#_of_views,2), in this case (1,22,4,2)
    Proj_points = np.stack(filered_corners,axis = 2)
    Proj_Mat = np.stack(ProjectionMats,axis=0)
    
    #stack projection matrix and pixel coordinates for bundle adjustment
    BA_points2D = np.stack(filered_corners,axis = 0)
    
    flatten_ProjectionMats = []
    for P in ProjectionMats:
        flatten_ProjectionMats.append(P.ravel())

    input_param = np.hstack(flatten_ProjectionMats)
    
    coords = triangulate(Proj_points,Proj_Mat).solveA()#tranigulate points
    coords = coords[:,:,:-1]

    VIS(coords)

    input_points = coords.reshape((-1,))
    ba_input = np.hstack((input_points,input_param))

    refined_coords, refined_Pmat= bundle_adjustment(1,BA_points2D,ba_input,calibration_points)

    
    l = len(refined_Pmat)//cam_number
    #split the optimized projection matrix
    ret = []
    for i in range(cam_number):
        j = i+1
        P = refined_Pmat[i*l:j*l].reshape((3,4))
        ret.append(P)

    #returns projection matrix/camera positions/3d coordinates of charucoboard corners  
    return Proj_Mat,camera_positions,coords



def run(ProjectMatix):
    
    #=================load skeleton 2d keypoints,stack them/create a dictionary according to camera order
    pixelCoord_path = os.listdir(Pixel_coord_FIlE_PATH)
    path = []
    for p in pixelCoord_path:
        if not p.startswith('.'):
            path.append(p)
    path.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    PixelCoord = {}

    for i in range(len(path)):
        PixelCoord[i] = np.load(Pixel_coord_FIlE_PATH+'/'+path[i],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:,:]


    Proj_points = []
    for key in PixelCoord.keys():
        Proj_points.append(PixelCoord[key])
    
    Proj_points = np.array(Proj_points)
    Proj_points = np.stack(Proj_points,axis = 2)
    Proj_Mat = np.stack(ProjectMatix,axis=0)


    coords = triangulate(Proj_points,Proj_Mat).solveA()#tranigulate points
    coords = coords[:,:,:-1]
    Vis(VideoFolder+'/'+Source_video_List[0][0],VideoFolder+'/'+Source_video_List[1][0],VideoFolder+'/'+Source_video_List[2][0],VideoFolder+'/'+Source_video_List[3][0],coords).display()

    return coords



if __name__ == '__main__':
    
     #######input video(raw data), fill the entry RawVideo with the path to the raw video file 

    
    #1.preprocess video (concat trim....),save them in VideoFolder ('data/SourceVideos')

    
    #3.run openpose, save npy files in the 'data/OpenPose'

    
    #run create calibration function, this function set up all necessary data for calibration
    create_calibration()

    #calibrate the cameras
    P,Cam_Pos,charuco_coords = calibration('Calibration')
    
    #3D calculation 
    coords = run(P)

