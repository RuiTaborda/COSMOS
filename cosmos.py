# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:38:27 2019

@author: rui taborda
# https://ciencias.ulisboa.pt/pt/perfil/rtaborda

"""

import cv2
import numpy as np
from xml.dom import minidom
import glob
import os
import pandas as pd 
from pathlib import Path
import sys
import av

class ROI:
    xori = -112000
    yori = -77571
    rotation = 50
    pixel_size = 0.11
    dx_map = 600
    dy_map = 300
    
    def __init__(self, *initial_data,  **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dx = int(self.dx_map / self.pixel_size)
        self.dy = int(self.dy_map / self.pixel_size)
 
    
    def xyz2XYZ(self, xyz):  #from local xyz to global coordinates XYZ
        z = np.c_[xyz[:,2]]
        ones = np.ones(shape=(len(xyz), 1))
        xy1 = np.hstack([xyz[:,:2], ones])
        atm = self.affine_trans_mat()
        XYZ = atm.dot(xy1.T).T
        XYZ = np.hstack([XYZ, z * self.pixel_size])
        XYZ[:,0] = XYZ[:,0] + self.xori
        XYZ[:,1] = XYZ[:,1] + self.yori
        return XYZ
    
    def XYZ2xyz(self, points):  #from global XYZ to local coordinates xyz 
        XYZ = points.copy()
        XYZ[:,0] = XYZ[:,0] - self.xori
        XYZ[:,1] = XYZ[:,1] - self.yori
        Z = np.c_[XYZ[:,2]] / self.pixel_size
        ones = np.ones(shape=(len(XYZ), 1))
        XY1 = np.hstack([XYZ[:,:2], ones])
        rot_mat = cv2.getRotationMatrix2D((0, 0), self.rotation, 1 / self.pixel_size)

        xyz = rot_mat.dot(XY1.T).T
        xyz = np.hstack([xyz, Z])
        return xyz
    
    def affine_trans_mat(self):  #from global XYZ to local coordinates xyz 
        return cv2.invertAffineTransform(cv2.getRotationMatrix2D((0, 0), self.rotation, 1 / self.pixel_size))
    
    def save_world_file(self, filename):
        atm = self.affine_trans_mat()
        atm[0,2] = self.xori
        atm[1,2] = self.yori
        if Path(filename).suffix =='.jpg':
            file = filename.replace('.jpg', '.jpw')
        else:
            sys.exit("Unknow file type")
        np.savetxt(file, atm.T.flatten())

class VideoImage:
    camera_file = 'D:/Documentos/Modelos/Rectify/tutorial/Exercise/Cameras.xml'
    camera = 'Mobo10_4_225_136'
    corrected_images = True
    camera_position = []
    camera_rot = []

    gcp_excel_file = ''
    
    gcp_XYZ = []
    gcp_uv = []
    H = np.array([]) # homography matrix
    z_plane = 5

    original_images_dir = 'D:/Documentos/Modelos/Rectify/tutorial/Exercise/ExtParDefinition/GCP_08Out2010/'
    images_filename_extension = 'jpg'
    images = ''
    image_extensions = ('png', 'jpg', 'jpeg', 'tif')
    video_extensions = ('3gp', 'avi')
    
    image_display = True
    
    homography_matrix_filename = []
    rectification_method = 'from_gcp'
    
    
    undistort_images_dir = 'D:/Documentos/Modelos/Rectify/'
    undistort = True
    write_echo = True
    write_world_file = True
    
    
    #interpolation = cv2.INTER_LINEAR
    interpolation_method = cv2.INTER_NEAREST
    
    rectified_images_dir = 'D:/Documentos/Modelos/Rectify/rectified/'
    rectify = False
    RANSAC_scheme = True
    
    roi = ROI()
   
    alpha = 0;
    mtx = np.zeros((3,3))
    dist = np.zeros((1,5))
    newcameramtx = np.zeros((3,3))
    mtx = np.zeros((3,3))

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

       
   
    def undistort_images(self):
         self.read_camera()
 
         self.read_images(self.original_images_dir)
         Path(self.undistort_images_dir).mkdir(parents=True, exist_ok=True)
         for fname in self.images:
             
            # Undistort images
             
            if fname.lower().endswith(self.image_extensions):
                img = cv2.imread(fname)
                dst = self.undistort(img)
                basename = os.path.basename(fname)
                img_name = self.undistort_images_dir + basename
                writeStatus = cv2.imwrite(img_name, dst)
                
                if self.image_display:
                    cv2.imshow('Undistorted image', dst)
                if self.write_echo:
                    if writeStatus is True:
                        print("Undistort image " +  img_name + " written")
                    else:
                        print("Problem written " + img_name) #
           
            # Undistort videos
            

            
            
            elif fname.lower().endswith(self.video_extensions):
                print(fname)
                container = av.open(fname)

                for frame in container.decode(video=0):
                    if frame.index % 100 == 0:
                        print("processed frame index {}".format(frame.index))

                    img = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                    dst = self.undistort(img)
                    if frame.index == 0:
                        h,  w = dst.shape[:2]
                        basename = os.path.splitext(os.path.basename(fname))[0] + '.avi'
                        vid_name = self.undistort_images_dir + basename
                        out = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc('M','J','P','G'), 2, (w,h))
                    out.write(dst)
                        
                    if self.image_display:
                        cv2.imshow('Frame', dst) 
                        # Press Q on keyboard to  exit 
                        if cv2.waitKey(1) & 0xFF == ord('q'): 
                            break
                out.release()
                
            else:
                print('unknown file extension')



              
    def rectify_images(self, method = 'from_gcp'):
         if self.rectification_method == 'from_gcp':
             self.read_camera()
             self.dist = np.zeros((1,5))
             self.compute_camera_matrices()
         elif self.rectification_method == 'from_parameter_file':
             parameters = np.load(self.rectification_parameter_file, allow_pickle=True)
             self.H = parameters['H']
             self.z_plane = parameters['z_plane']
             self.roi = ROI(parameters['roi'].all())
             
         if not self.H.size:
             sys.exit('Empty Homography Matriz - cannot rectify images')
         self.read_images(self.undistort_images_dir)
         Path(self.rectified_images_dir).mkdir(parents=True, exist_ok=True)
         for fname in self.images:
             print(fname)
             if fname.lower().endswith(self.image_extensions):
                img = cv2.imread(fname)
                dst = self.warp_image(img)
                basename = os.path.basename(fname)
                img_name = self.rectified_images_dir + basename
                writeStatus = cv2.imwrite(img_name, dst)
                if self.write_world_file:
                    self.roi.save_world_file(img_name)
                if self.write_echo:
                    if writeStatus is True:
                        print("Rectified image " +  img_name + " written")
                    else:
                        print("Problem written " + img_name) #


             elif fname.lower().endswith('avi'):  #only rectifies avi files
                    print(fname)
                    container = av.open(fname)

                    for frame in container.decode(video=0):
 
                        img = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                        dst = self.warp_image(img)
                        
                        if frame.index % 100 == 0:
                            print("processed frame index {}".format(frame.index))
                               
                            if self.image_display:
                                cv2.imshow('Frame', dst) 
                                # Press Q on keyboard to  exit 
                                if cv2.waitKey(1) & 0xFF == ord('q'): 
                                    break
                        
                        if frame.index == 0:
                            h,  w = dst.shape[:2]
                            basename = os.path.splitext(os.path.basename(fname))[0] + '.avi'
                            vid_name = self.rectified_images_dir + basename
                            out = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc('M','J','P','G'), 2, (w,h))
                        out.write(dst)
                    out.release()
   
             else:
                    print('unknown file extension')
     


                
    def warp_image_points(self, img):
         h,  w = img.shape[:2]
         u, v = np.meshgrid(np.arange(w),np.arange(h))
         uv = np.vstack((u.flatten(), v.flatten())).T
         xy = self.uv2xyz(uv.astype(np.float32))
         x = xy[:,0]
         y = xy[:,1]
         
         ind = (x > self.roi.xori) & (x < self.roi.xori + self.roi.dx) & (y > self.roi.yori) & (y < self.roi.yori + self.roi.dy)
         
         x = x[ind]
         y = y[ind]
         
         ptsRGB=img.reshape(-1,3)
         
         ptsRGB = ptsRGB[ind,:]
         
         return x,y, ptsRGB
 
    def warp_image(self, img):
        warp = cv2.warpPerspective(img, self.H, (self.roi.dx, self.roi.dy), flags=cv2.WARP_INVERSE_MAP + self.interpolation_method)
        return warp
    
    def uv2XYZ(self, points):
        Hinv = np.linalg.inv(self.H)
        xy = cv2.perspectiveTransform(np.array([points]), Hinv)[0]
        z = self.z_plane * np.ones(shape=(len(xy), 1))
        xyz = np.hstack([xy[:,:2], z])
        XYZ = self.roi.xyz2XYZ(xyz)
        return XYZ
   
    def XYZ2uv(self, points):
        points = self.roi.XYZ2xyz(points)
        points = points[:,:2]
        uv = cv2.perspectiveTransform(np.array([points]), self.H)[0]
        return uv
            
    def undistort(self, img):
        h,  w = img.shape[:2]
        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), self.alpha,(w,h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        return dst
    
    def read_images(self, images_dir):
        self.images = glob.glob(images_dir + '*.' + self.images_filename_extension)
        
    def compute_camera_matrices(self):
        self.read_gcp()
        xyz = self.roi.XYZ2xyz(self.gcp_XYZ)
        if self.RANSAC_scheme:
            retval, self.rvec, self.tvec, _ = cv2.solvePnPRansac(xyz, self.gcp_uv, self.mtx,  self.dist)
        else:
            retval, self.rvec, self.tvec = cv2.solvePnP(xyz, self.gcp_uv, self.mtx,  self.dist)
        self.camera_rot = cv2.Rodrigues(self.rvec)[0]
        camera_position = -self.camera_rot.T @ self.tvec
        self.camera_position = self.roi.xyz2XYZ(camera_position.T)
        
        Rt = self.camera_rot
        Rt[:,2] = Rt[:,2] * self.z_plane / self.roi.pixel_size + self.tvec.flatten()
        self.H = self.mtx @ Rt
        self.H = self.H / self.H[2,2]
        
    def save_rectification_parameter_file(self, filename = 'rectification_parameters'):
        np.savez(filename, H = self.H, z_plane = self.z_plane, roi = self.roi.__dict__)
    
    def gcp_reprojection_error(self):
        #error in pixels
        dif = self.gcp_uv - self.XYZ2uv(self.gcp_XYZ)
        return np.linalg.norm(dif, axis=1)
    
    def read_gcp(self):
        gcp = pd.read_excel(self.gcp_excel_file)
        self.gcp_XYZ=gcp[['X', 'Y', 'Z']].values.astype('float64')
        self.gcp_uv=gcp[['u', 'v']].values.astype('float64')

    def read_camera(self):
        self.mtx = np.zeros((3,3))
        xmldoc = minidom.parse(self.camera_file)
        itemlist = xmldoc.getElementsByTagName('Cameras')
        camera_par = itemlist[0].getElementsByTagName(self.camera)
        fx = camera_par[0].getElementsByTagName('fx')
        self.mtx[0, 0] = fx[0].firstChild.data 
        fy = camera_par[0].getElementsByTagName('fy')
        self.mtx[1, 1] = fy[0].firstChild.data 
        cx = camera_par[0].getElementsByTagName('cx')
        self.mtx[0, 2] = cx[0].firstChild.data
        cy = camera_par[0].getElementsByTagName('cy')
        self.mtx[1, 2] = cy[0].firstChild.data
        self.mtx[2, 2] = 1
        k1 = camera_par[0].getElementsByTagName('k1')
        self.dist[0, 0] = k1[0].firstChild.data
        k2 = camera_par[0].getElementsByTagName('k2')
        self.dist[0, 1] = k2[0].firstChild.data
        k3 = camera_par[0].getElementsByTagName('k3')
        self.dist[0, 2] = k3[0].firstChild.data
        k4 = camera_par[0].getElementsByTagName('k4')
        self.dist[0, 3] = k4[0].firstChild.data
        k5 = camera_par[0].getElementsByTagName('k5')
        self.dist[0, 4] = k5[0].firstChild.data
        
	