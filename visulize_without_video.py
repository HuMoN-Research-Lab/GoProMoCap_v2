import numpy as np
import sys
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

#Chess = np.load('CJC4Chess.npy')

class Vis:
    """
    video_path:list of video path
    coords: a set of 3D coordinates with shape (# of frames,#of keypoints,3Dcoords)
    """
    def __init__(self,left_video_path,right_video_path,mid_video_path,fouth_cam_path,coords):
        self.leftPath = left_video_path
        self.rightPath = right_video_path
        self.midPath = mid_video_path
        self.f_path = fouth_cam_path
        self.coords = coords
        #self.pixelCoord = pixelCoord

    
    def display(self):
        # Create a VideoCapture object and read from input file 
        leftCap = cv2.VideoCapture(self.leftPath)
        rightCap = cv2.VideoCapture(self.rightPath)       
        midCap = cv2.VideoCapture(self.midPath)
        fouthCap = cv2.VideoCapture(self.f_path)


        np.random.seed(19680801)


        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        
        writer = FFMpegWriter(fps=30, metadata=metadata)
        
        
        #3D coordinates data and pixel data
        d = self.coords

        #create subplots
        fig = plt.figure()
        ax1 = plt.subplot(2,3,1,projection='3d') #3D



        ax2 = plt.subplot(2,3,2) #4th cam
        ax3 = plt.subplot(2,3,3) #left cam
        ax4 = plt.subplot(2,3,4) #right cam
        ax5 = plt.subplot(2,3,5) #mid cam


        #set up 3d coord data


        def grab_frame(cap):
            ret,frame = cap.read()
            return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #return frame
        

        plt.ion()
        with writer.saving(fig, "writer_test.mp4", 100):
            c = 0
            while c < d.shape[0]:
                #chessboard  = ax1.scatter3D(Chess[0,:54,0], Chess[0,:54,1], Chess[0,:54,2], marker='o', cmap='hot',s=20)
                temp = ax1.scatter3D(d[c,:67,0], d[c,:67,1], d[c,:67,2], marker='o', cmap='hot',s=20)
                temp2 = ax1.scatter3D(d[c,67:,0], d[c,67:,1], d[c,67:,2], marker='o', cmap='hot',s=100)
            
                im2, = ax1.plot(d[c,0:5,0],d[c,0:5,1],d[c,0:5,2],color = 'r',linewidth=2.0)
                im3, = ax1.plot(d[c,8:12,0],d[c,8:12,1],d[c,8:12,2],color = 'g',linewidth=2.0)
                im4, = ax1.plot([d[c,1,0],d[c,8,0]],[d[c,1,1],d[c,8,1]],[d[c,1,2],d[c,8,2]],color = 'r',linewidth=2.0)
                im5, = ax1.plot([d[c,1,0],d[c,5,0]],[d[c,1,1],d[c,5,1]],[d[c,1,2],d[c,5,2]],color = 'r',linewidth=2.0)
                im6, = ax1.plot(d[c,5:8,0],d[c,5:8,1],d[c,5:8,2],color = 'r',linewidth=2.0)
                im7, = ax1.plot([d[c,8,0],d[c,12,0]],[d[c,8,1],d[c,12,1]],[d[c,8,2],d[c,12,2]],color = 'g',linewidth=2.0)
                im8, = ax1.plot(d[c,12:15,0],d[c,12:15,1],d[c,12:15,2],color = 'g',linewidth=2.0)
                im9, = ax1.plot([d[c,11,0],d[c,24,0]],[d[c,11,1],d[c,24,1]],[d[c,11,2],d[c,24,2]],color = 'g',linewidth=2.0)
                im10, = ax1.plot([d[c,14,0],d[c,19,0]],[d[c,14,1],d[c,19,1]],[d[c,14,2],d[c,19,2]],color = 'g',linewidth=2.0)
                im11, = ax1.plot([d[c,14,0],d[c,21,0]],[d[c,14,1],d[c,21,1]],[d[c,14,2],d[c,21,2]],color = 'g',linewidth=2.0)
                im12, = ax1.plot([d[c,11,0],d[c,22,0]],[d[c,11,1],d[c,22,1]],[d[c,11,2],d[c,22,2]],color = 'g',linewidth=2.0)
                im13, = ax1.plot([d[c,22,0],d[c,23,0]],[d[c,22,1],d[c,23,1]],[d[c,22,2],d[c,23,2]],color = 'g',linewidth=2.0)
                im14, = ax1.plot([d[c,16,0],d[c,18,0]],[d[c,16,1],d[c,18,1]],[d[c,16,2],d[c,18,2]],color = 'r',linewidth=2.0)

            
                writer.grab_frame()
                plt.pause(0.0001)
                temp.remove()
                temp2.remove()
                im2.remove()
                im3.remove()
                im4.remove()
                im5.remove()
                im6.remove()
                im7.remove()
                im8.remove()
                im9.remove()
                im10.remove()
                im11.remove()
                im12.remove()
                im13.remove()
                im14.remove()


                c+=1
        
            plt.ioff() 
            plt.show()