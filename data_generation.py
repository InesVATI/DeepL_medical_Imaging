# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:58:13 2022

@author: INES
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pyeit.mesh as mesh
from pyeit.eit.utils import eit_scan_lines
from pyeit.eit.fem import Forward
from pyeit.mesh.shape import thorax
import matplotlib.pyplot as plt

import random as rd

  
def isintriangle(p1, p2, p3, p):
    """
    Enable to know either the point is inside the triangle defined by the vertices
    by computing the Barycentric coordinates 
    The method is described in : https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Determining_location_with_respect_to_a_triangle 
    
    Params
    -------
    p: point array 2D shape
    p1, p2, p3: vertices of the triangle numerotated in anticlockwise order
    
    Returns
    -------
    bool
        
    """
    
    detT = (p2[1]-p3[1])*(p1[0]-p3[0]) + (p3[0]-p2[0])*(p1[1]-p3[1])
    c1 = ((p2[1]-p3[1])*(p[0]-p3[0]) + (p3[0]-p2[0])*(p[1]-p3[1]))/detT
    if c1 < 0:
        return False
    
    c2 = ((p3[1]- p1[1])*(p[0]-p3[0])+ (p1[0] -p3[0])*(p[1]-p3[1]))/detT
    if c2 <0:
        return False
    
    c3 = 1 - c1- c2
    if c3<0:
        return False
    
    return True
          

def label_data_rand(num, nb_px = 128, plot=False):
    """"
    Compute and save the conductivity map and the voltage at the borders
    Params:
    -----------
    num: int used for the name of the file
    nb_px: int 
    
    """
    
    #set the parameters of the random anomaly
    ray = (0.7-0.04)*rd.random() + 0.04 #RAY randomly between 0.04 and 0.7 
    #print('ray is ', ray)
    perm = (60000- 500)*rd.random() + 500
    #print('perm is ', perm)
    #x in [-1, 1] , y in [-0.6, 0.6]
    ano_x, ano_y = 2*rd.random() -1, 1.2*rd.random()- 0.6
    
    #conductivity is equal to 1 every where
    mesh_obj, el_pos = mesh.create(16, h0=0.1, fd=thorax)
    # extract node, element, alpha
    pts = mesh_obj["node"]  # array of coordinates (x,y) for each mesh's point 
    #For each triangle, the indices of the three points that make up the triangle, ordered in an anticlockwise manner.
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]
    
    #add one random anomaly 
    anomaly = [{"x": ano_x, "y": ano_y, "d": ray, "perm": perm}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)
    
    delta_perm = mesh_new["perm"] - mesh_obj["perm"] #match each triangle to a conductivity value
    
    """ 1. Save the array containing the conductivity """
    cond_img = np.zeros((nb_px, nb_px))
    
    #graduation for pixelization of the image
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    grad_x = (xmax-xmin)/nb_px
    grad_y = (ymax-ymin)/nb_px
    
    for i in range(nb_px):
        for j in range (nb_px):
            
            # the pixel take the value of the conductivity of the triangle 
            #in which its center is 
            center = [(2*xmin + (2*i +1)*grad_x)/2, (2*ymin + (2*j +1)*grad_y)/2]
            for k in range (tri.shape[0]):
                p1 = [x[tri[k, 0]], y[tri[k,0]]]
                p2 = [x[tri[k, 1]], y[tri[k,1]]]
                p3 = [x[tri[k, 2]], y[tri[k,2]]]
                
                if isintriangle(p1, p2, p3, center):
                    cond_img[i,j] = delta_perm[k]
                    break 

    if plot:
        fig, ax = plt.subplots(1, constrained_layout=True)
        fig.set_size_inches(6, 4)
        
        ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
        ax.set_aspect("equal")
        plt.show()
        
    #return cond_img
    np.save('labeled_data/'+str(num)+'_cond_img', cond_img)
    
    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    # adjacent stimulation (el_dist=1), adjacent measures (step=1)
    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(16, el_dist)
    
    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    np.save('labeled_data/'+str(num)+'_voltage_border',f1.v - f0.v)
    


def test(num=0):
    label_data_rand(num)