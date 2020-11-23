import numpy as np
import sys
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import cv2.aruco as aruco
from config import video_resolution,dataFile,num_of_cameras
from scipy.optimize import least_squares
import time
from scipy.sparse import lil_matrix


base_cam = {'A':0,'B':1,'C':2,'D':3}

class Exceptions(Exception):
    pass


#============export a csv file
def toCsv(coord,name):
    np.savetxt(name,coord,delimiter=",",fmt='%10.5f')


#============convert a vector to its skewed symmetric form
def vec2skewMat(vec):
    res = np.zeros((3,3))
    res[0,1],res[0,2],res[1,0],res[1,2],res[2,0],res[2,1] = -vec[2],vec[1],vec[2],-vec[0],-vec[1],vec[0]

    return res

#=============combine rotation and translation vector to an homogenies transformation matrix
def R_t2H(R,T):
    ret = np.zeros((4,4))
    ret[:3,:3],ret[:3,-1] = R,T.reshape(3,)
    ret[-1,:] = np.array([0,0,0,1])

    return ret



#___________inverse a homogenies transformation matrix
def inverseH(H):
    trans_R = H[:3,:3].T#why transpose
    trans_T = -trans_R.dot(H[:3,-1])
    inv_H = np.zeros((4,4))
    inv_H[:3,:3] = trans_R
    inv_H[:3,-1] = trans_T
    inv_H[-1,-1] = 1

    return inv_H


def get_TransMat(H_mats):
    
    """
    input: set of 4x4 homogenies transformation matrixs(charuco board is the original)
    output: list of  3x4 projection matrixs, has same length with the input

    calculate homogeneous transformation matrix between base Cam and the other cameras
    and calculate the projection matrix of all cameras
    """

    if num_of_cameras != len(H_mats):
        raise Exceptions('number of homogenies transformation matrixs must equal to number of cameras')

    #the first homogenies transformation matrixs is used to calculate the relative position to the principle camera(first camera)
    H0 = H_mats[0]
    #the principle camera have projection matrix with rotation matrix to be a diagonal matrix, translation vector to be (0,0,0)
    M_base = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    ret = []
    ret.append(M_base)

    #calculate each cameras relative position to the principle camera
    for i in range(1,num_of_cameras):
        H_temp = H_mats[i]
        #H is the relative postion of camera i to the principle camera
        H = H_temp.dot(inverseH(H0))
        #extract rotation and translation vector from the calculated projection matrx
        T,R = H[:3,-1].reshape(3,1),H[:3,:3]
        M = np.hstack((R, T))

        ret.append(M)

    assert len(ret) == num_of_cameras
    
    return ret


#calibration funciton(chessboard)
def get_RT_mtx(path,Cam_indx,video_resolution):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)*25

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(path)
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    count = 0
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img,video_resolution)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners   
        ret, corners = cv2.findChessboardCorners(gray, (6,9),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(50)
    
        count += 1
        print(count)

    cv2.destroyAllWindows()

    #=================store camera information
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_ret.npy',ret)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_mtx.npy',mtx)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_dist.npy',dist)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_rvec.npy',rvecs)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_tvecs.npy',tvecs)

    return ret,mtx,dist,rvecs,tvecs





def video_loader(video_path):
    files = os.listdir(video_path)
    path = []
    for p in files:
        if not p.startswith('.'):
            path.append(p)

    path.sort(key=lambda f: int(re.sub('\D', '', f)))#sort path
    print(path)
    
    for i in range(len(path)):
        vidcap = cv2.VideoCapture(os.path.join(video_path,path[i]))
        success,image = vidcap.read()
        count = 0

        while success:
            success,image = vidcap.read()
            if success:
                height , width , layers =  image.shape
                resize = cv2.resize(image, video_resolution) 
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #single_video.append(image)
                if count < 2:
                    cv2.imwrite("Calibration/"+str(i+1)+"/frame%d.jpg" % count, resize)     # save frame as JPEG file
                    print(resize.shape)
                else:
                    break
                
                print('Read a new frame: ', success)
                count += 1
                print(count)
            
            else:
                break

def video_loader_old(fileName,Cam_Indx):
    """
    Cam_Index: CamA/CamB/CamC, depand on how many cameras are used during recording
    """

    DATADIR_1 = dataFile+'/Synced'
    datadir =[DATADIR_1]
    video_array = []
    for dir in datadir:
        for video in os.listdir(dir):
            if video == fileName:
                vidcap = cv2.VideoCapture(os.path.join(dir,video))
                success,image = vidcap.read()
                count = 0
                #success = True

                while success:
                    success,image = vidcap.read()
                    if success:
                        height , width , layers =  image.shape
                        resize = cv2.resize(image, video_resolution) 
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #single_video.append(image)
                        if count < 2:
                            cv2.imwrite("Calibration/"+Cam_Indx+"_Calibration/frame%d.jpg" % count, resize)     # save frame as JPEG file
                            print(resize.shape)
                        else:
                            break

    
                        print('Read a new frame: ', success)
                        count += 1
                        print(count)
                    else:
                        break



#=======================triangulate points
class triangulate:
    """
    ImgPoints: a (frame,#_of_keypoints,#_of_views,3) matrix -> (x,y,prob)
    ProjectMat: a Mx3x4 matrix, M is number of views, each views has its 3x4 projection matrix
    """
    
    def __init__(self,ImgPoints,ProjectMat):
        
        self.ImgPoints = ImgPoints
        self.ProjectMat = ProjectMat

    def solveA(self): 

        if self.ImgPoints.shape[2] != len(self.ProjectMat):
            raise Exceptions('number of views must be equal to number of projection matrix')
        
        N_views = len(self.ProjectMat)
        A = np.zeros((N_views*2,4)) #prepare svd matrix A
        X = np.zeros((self.ImgPoints.shape[0],self.ImgPoints.shape[1],4))

        for i in range(self.ImgPoints.shape[0]): #for each point
            
            for k in range(self.ImgPoints.shape[1]):
                #if the lowest p-value of the obersvations is less than 0.9, skip the calculation
                if float(min(list(self.ImgPoints[i,k,:,-1]))) < -1:
                    continue
                for j in range(N_views): #for each view
                    
                    u,v = self.ImgPoints[i,k,j,0],self.ImgPoints[i,k,j,1] #initialize x,y points
                    
                    for col in range(4):
                        A[j*2+0,col] = u*self.ProjectMat[j,2,col] - self.ProjectMat[j,0,col]
                        A[j*2+1,col] = v*self.ProjectMat[j,2,col] - self.ProjectMat[j,1,col]
     

                U,s,V = np.linalg.svd(A)
                P = V[-1,:] / V[-1,-1]
                #X[i] = P[:-1]
                X[i,k] = P
        
        return X


#=====================triangulation pick best angle version 
class triangulateTest:
    """
    ImgPoints: a (frame,#_of_keypoints,#_of_views,3) matrix -> (x,y,prob)
    ProjectMat: a Mx3x4 matrix, M is number of views, each views has its 3x4 projection matrix
    """
    
    def __init__(self,ImgPoints,ProjectMat,base_cam):
        
        self.ImgPoints = ImgPoints
        self.ProjectMat = ProjectMat
        self.base_cam = base_cam
    

    def solveA(self): 

        if self.ImgPoints.shape[2] != len(self.ProjectMat):
            raise Exceptions('number of views must be equal to number of projection matrix')
        
        N_views = len(self.ProjectMat)
        A = np.zeros((N_views*2,4)) #prepare svd matrix A
        X = np.zeros((self.ImgPoints.shape[0],self.ImgPoints.shape[1],4))
        #vis_list = {}
        vis_list = []
        
        for i in range(self.ImgPoints.shape[0]): #for each point
            
            for k in range(self.ImgPoints.shape[1]):
                
                v_indx,max_ = -1.0,-1.0
                for view_index in range(self.ImgPoints.shape[2]):
                    if view_index == self.base_cam:
                        continue
                    elif self.ImgPoints[i,k,view_index,2] > max_:
                        max_ = self.ImgPoints[i,k,view_index,-1]
                        v_indx = view_index
                
                #vis_list[(i,k)] = v_indx
                vis_list.append(v_indx)
                
                view_list = [self.base_cam,v_indx]
                
                for j in view_list: #for each view            
                    u,v = self.ImgPoints[i,k,j,0],self.ImgPoints[i,k,j,1] #initialize x,y points
                    
                    for col in range(4):
                        A[j*2+0,col] = u*self.ProjectMat[j,2,col] - self.ProjectMat[j,0,col]
                        A[j*2+1,col] = v*self.ProjectMat[j,2,col] - self.ProjectMat[j,1,col]
        
                A1 = np.zeros((4,4))
                A1[0,:] = A[view_list[0]*2,:]
                A1[1,:] = A[view_list[0]*2+1,:]
                A1[2,:] = A[view_list[1]*2,:]
                A1[3,:] = A[view_list[1]*2+1,:]
                A1 = np.array(A1)
                A1 = A1.reshape((4,4))
                U,s,V = np.linalg.svd(A)
                P = V[-1,:] / V[-1,-1]
                #X[i] = P[:-1]
                X[i,k] = P
        
        return X,vis_list



#============================calibration using single aruco marker

def aruco_detect(path,Cam_indx,video_resolution):

    if not import_camera_intrinsic:
        _,K,dist,_,_ = get_RT_mtx(path,Cam_indx,video_resolution)
    else:
        K = np.load('imported_Camera_INFO/CamA_mtx.npy')
        dist = np.load('imported_Camera_INFO/CamA_dist.npy')

    images = glob.glob(path)
    count = 0
    for fname in images:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        img = cv2.imread(fname)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # detector parameters can be set here (List of detection parameters[3])
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        rvec_d = {}
        tvec_d = {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        if np.all(ids != None):

            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(corners,0.1, K,dist)
            if count == 0:
                for i in range(len(ids)):
                    #ret[ids[i]] = {'rvec':list(rvec[i].reshape((-1))),'tvec':list(tvec[i].reshape((-1)))}
                    rvec_d[int(ids[i])] = rvec[i].reshape((-1)).tolist()
                    tvec_d[int(ids[i])] = tvec[i].reshape((-1)).tolist()

                # np.save('ARUCO_POSE/CamA_t.npy',tvec)
                # np.save('ARUCO_POSE/CamA_r.npy',rvec)

            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            for i in range(0, ids.size):
                # draw axis for the aruco markers
                cv2.aruco.drawAxis(img, K, dist, rvec[i], tvec[i], 0.1)

            # draw a square around the markers
            cv2.aruco.drawDetectedMarkers(img, corners)


            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(img, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            count += 1

            cv2.imshow('frame',img)
            key = cv2.waitKey(3000)

            cv2.destroyAllWindows()



            return K,dist,rvec_d,tvec_d
    
        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)




#=================calibration using charucoboard
def charuco_detect(path,video_resolution):
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    images = glob.glob(path)



    def read_chessboards(images):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in images:
            print("=> Processing image {0}".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(corners)>0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(gray, corner,
                                    winSize = (5,5),
                                    zeroZone = (-1,-1),
                                    criteria = criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
                if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

            decimator+=1

        imsize = gray.shape
        return allCorners,allIds,imsize
    
    def calibrate_camera(allCorners,allIds,imsize):
        """
        Calibrates the camera using the dected corners.
        """
        print("CAMERA CALIBRATION")

        cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                    [    0., 1000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        #flags = (cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients0,
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners,
                        charucoIds=allIds,
                        board=board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    allCorners,allIds,imsize=read_chessboards(images)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)
    tvecs = np.array(tvecs)
    mtx = np.array(mtx)
    rvecs = np.array(rvecs)
    

    return mtx,dist,rvecs[0],tvecs[0],allCorners[0],allIds[0]





#sparse bundle adjustment(stable)
def SBA(Len_of_frame,points2d,ba_input,VIS_cam_List,points_inFrame,base_Cam_Index):
    """
    Len_of_points:how many points to be recontrusct
    points2D: all pixel locations(ground turth)
    ba_input:1D vector include flattened 3d points and projection matrix
    points_inFrame:number of each keypoints per frame
    base_Cam_Index: 'A'/'B'/'C'
    """

    def fun(ba_input):
        p = ba_input[:Len_of_frame*3*points_inFrame].reshape((-1,points_inFrame,3)) #reshape back to(len,25,3)
        param = ba_input[Len_of_frame*points_inFrame*3:]

        temp = np.ones((p.shape[0],p.shape[1],1))
        x = np.concatenate((p,temp),axis=2)
        true_pixel_coord = np.zeros((2*Len_of_frame,points_inFrame,2))

        if num_of_cameras == 2:
            l = len(param)//2
            ProjMats = (param[:l].reshape((3,4)),param[l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]
            
            
        elif num_of_cameras == 3:
            l = len(param)//3
            ProjMats = (param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]
        

        elif num_of_cameras == 4:
            l = len(param)//4
            ProjMats = (param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:3*l].reshape((3,4)),param[3*l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]

            
            
        reproj_points = np.vstack((reproj1,reproj2))
 
        reproj_points = reproj_points[:,:,:2] / reproj_points[:,:,2,np.newaxis]
        res = (reproj_points-true_pixel_coord)**2
        #res = abs(reproj_points-true_pixel_coord)


        return res.ravel()


    def bundle_adjustment_sparsity(n_point3D):
        """
        n_observation:total length of pixel coordinates

        """
        m = n_point3D * 2 * 2 #row
        n = n_point3D * 3 + 12*num_of_cameras #col 
        A = lil_matrix((m, n), dtype=int)

        if num_of_cameras == 2:
            if base_Cam_Index == 'A':
                A[:m//2,-24:-12] = 1
                A[m//2:,-12:] = 1
            else:
                A[m//2:,-24:-12] = 1
                A[:m//2,-12:] = 1
        
        elif num_of_cameras == 3:
            if base_Cam_Index == 'A':
                A[:m//2,-36:-24] = 1
            elif base_Cam_Index == 'B':
                A[:m//2,-24:-12] = 1
            elif base_Cam_Index == 'C':
                A[:m//2,-12:] = 1
            
            for i in range(n_point3D):
                s1,s2 = (VIS_cam_List[i]-3)*12,(VIS_cam_List[i]-3)*12+12
                if s2 == 0:
                    A[2*i,s1:] = 1
                    A[2*i+1,s1:] = 1
                else:
                    A[2*i,s1:s2] = 1
                    A[2*i+1,s1:s2] = 1 
        
        elif num_of_cameras == 4:
            if base_Cam_Index == 'A':
                A[:m//2,-48:-36] = 1
            elif base_Cam_Index == 'B':
                A[:m//2,-36:-24] = 1
            elif base_Cam_Index == 'C':
                A[:m//2,-24:-12] = 1
            elif base_Cam_Index == 'D':
                A[:m//2,-12:] = 1
            
            for i in range(n_point3D):
                s1,s2 = (VIS_cam_List[i]-4)*12,(VIS_cam_List[i]-4)*12+12
                if s2 == 0:
                    A[2*i,s1:] = 1
                    A[2*i+1,s1:] = 1
                else:
                    A[2*i,s1:s2] = 1
                    A[2*i+1,s1:s2] = 1 
                    

        for i in range(n_point3D):
            for s in range(3):
                A[2*i,i*3+s] =1 
                A[2*i+1,i*3+s] =1
    

        A[m//2:,:-12*num_of_cameras] = A[:m//2,:-12*num_of_cameras]

        
        return A
    
    
    
    residual = fun(ba_input)
    print('max index',np.argmax(residual))
    A = bundle_adjustment_sparsity(Len_of_frame*points_inFrame)
    plt.plot(residual)
    plt.show()

    x0 = ba_input

    t0 = time.time()
    res = least_squares(fun,x0,jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    #res = least_squares(fun,x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()

    param = res.x
    print(param.shape)
    optimized_3D = param[:Len_of_frame*3*points_inFrame]
    Optimized_Proj_mat = param[Len_of_frame*3*points_inFrame:]

    coords = optimized_3D.reshape((-1,points_inFrame,3))

    return coords,Optimized_Proj_mat


#sparse bundle adjustment(working)
def SBA2(Len_of_frame,ProjMats,points2d,ba_input,VIS_cam_DIC,param):
    
    def fun(ba_input):
        p = ba_input[:Len_of_frame*3*points_inFrame].reshape((-1,points_inFrame,3)) #reshape back to(len,number of 3d points,xyz)
        #param = ba_input[Len_of_frame*points_inFrame*3:]
    
        temp = np.ones((p.shape[0],p.shape[1],1))
        x = np.concatenate((p,temp),axis=2) #attach 1 to the end of xyz, make it homogenis coordinates, the shape is (frame,#key points, xyz1)
        true_pixel_list = []
        reprojected_list = []

        if num_of_cameras == 2:
            l = len(param)//2
            ProjMats = (param[:l].reshape((3,4)),param[l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]
            
            
        elif num_of_cameras == 3:
            l = len(param)//3
            ProjMats = (param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:].reshape((3,4)))#it is stacked in the order which predefine by the principle camera
            
            #filtering through every points,frame 1 has point 0-66, frame 2 has points 67-66*2
            for point_id in VIS_cam_DIC.keys():
                for view_index in VIS_cam_DIC[point_id]:
                    frame_id = point_id // points_inFrame
                    point_index = point_id % points_inFrame

                    #convert the points index to format (frame,#_of_keypoints,#_of_views,3), we now able to locate the point but an additional step is to check if in this certain camera angle
                    # 'view_index', the pixel coordinates is reliable. -1 means it is not relibale,ow the index given by the view_index shows which camera it comes from and link the projection matrix

                    if view_index != -1:
                        true_pixel_coord = points2d[view_index,frame_id,point_index]
                        true_pixel_list.append(true_pixel_coord)
                        reprojected_coords = x[frame_id][point_index].dot(ProjMats[view_index].T)
                        #reprojected_coords = ProjMats[view_index].dot(x[frame_id][point_index])
                        reprojected_list.append(reprojected_coords)

        if len(true_pixel_list) != len(reprojected_list):
            raise Exceptions('unbalence equation')
        
        
        reprojected_list = np.array(reprojected_list)
        true_pixel_list = np.array(true_pixel_list)


        #reprojected_list = reprojected_list[:,:,:2] / reprojected_list[:,:,2,np.newaxis]
        reprojected_list = reprojected_list[:,:2] / reprojected_list[:,2,np.newaxis]
        res = abs(reprojected_list-true_pixel_list)

        return res.ravel()

    x0 = ba_input

    residual = fun(ba_input)
    
    plt.plot(residual)
    plt.show()

    t0 = time.time()
    res = least_squares(fun,x0,verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    #res = least_squares(fun,x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()

    param = res.x
    print(param.shape)
    optimized_3D = param[:Len_of_frame*3*points_inFrame]
    Optimized_Proj_mat = param[Len_of_frame*3*points_inFrame:]

    coords = optimized_3D.reshape((-1,points_inFrame,3))

    return coords,Optimized_Proj_mat
