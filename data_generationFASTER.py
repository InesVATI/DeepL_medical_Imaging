# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:58:13 2022

@author: INES
"""



#normaliser les images pour avec une norme égale à 1
#v_mes(sigma) = V_mes(lambda*sigma)/lambda à peut-être utiliser pour la normalisation





from __future__ import division, absolute_import, print_function
import numpy as np
import pyeit.mesh as mesh
from pyeit.eit.utils import eit_scan_lines
from pyeit.eit.fem import Forward
from pyeit.mesh.shape import thorax
import matplotlib.pyplot as plt
import time
import random as rd
import math

  
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

def center(xmin,ymax,grad_x,grad_y,i,j):
    return [(2*xmin + (2*j +1)*grad_x)/2, (2*ymax + (-2*i +1)*grad_y)/2]
          
def generate_array(nb_px, x, y, tri, delta_perm, test=False):
    """  Save the array of shape (nb_px, nb_px) containing the conductivity 
    from the caracteristics of the mesh 
    
    Params
    ---------
    nb_px: int, number of pixel
    x, y: array of shape (number of mesh's points, 1), stores the x-coordinate (respectively y-coordinate) of each point of the mesh
    tri: array of shape (number of triangles, 3), stores  the 3 vertices of each triangle
    test: bool, If True it plots the conductivity map and the triangulation image   
    
    Returns
    ---------
    array of shape (nb_px, nb_px)
    
    """
    
    cond_img = -5000*np.ones((nb_px, nb_px)) #the surrounding of the organ is set to -5000

    #graduation for pixelization of the image
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    grad_x = (xmax-xmin)/nb_px
    grad_y = (ymax-ymin)/nb_px
    # print("Nb de pixels: ",str(nb_px) + 'x' + str(nb_px) + ' =',nb_px*nb_px)
    # print("Nb de triangle: ",tri.shape[0])
    
    
    # for i in range(nb_px):
    #     for j in range (nb_px):
            
    #         # the pixel take the value of the conductivity of the triangle 
    #         #in which its center is 
    #         center = [(2*xmin + (2*j +1)*grad_x)/2, (2*ymax + (-2*i +1)*grad_y)/2]
    #         for k in range (tri.shape[0]):
    #             p1 = [x[tri[k, 0]], y[tri[k,0]]]
    #             p2 = [x[tri[k, 1]], y[tri[k,1]]]
    #             p3 = [x[tri[k, 2]], y[tri[k,2]]]
                
    #             if isintriangle(p1, p2, p3, center):
    #                 cond_img[i,j] = delta_perm[k]
    #                 break
    
    already_visited = np.zeros((nb_px,nb_px), dtype = bool)
                
    start2 = time.time()
    for indice_triangle in range(tri.shape[0]):
        p1 = [x[tri[indice_triangle, 0]] + abs(xmin), y[tri[indice_triangle,0]] + abs(ymin)] 
        p2 = [x[tri[indice_triangle, 1]] + abs(xmin), y[tri[indice_triangle,1]] + abs(ymin)] 
        p3 = [x[tri[indice_triangle, 2]] + abs(xmin), y[tri[indice_triangle,2]] + abs(ymin)] 
        
        p1_small = [p1[1]/grad_y,p1[0]/grad_x]#[p1[0]/grad_x,p1[1]/grad_y]
        p2_small = [p2[1]/grad_y,p2[0]/grad_x] #[p2[0]/grad_x,p2[1]/grad_y]
        p3_small = [p3[1]/grad_y,p3[0]/grad_x] #[p3[0]/grad_x,p3[1]/grad_y]
        
        #x_min = math.floor(min(p1_small[0],p2_small[0],p3_small[0]))
        x_min = math.floor(min(p1_small[1],p2_small[1],p3_small[1]))
        #y_min = math.floor(min(p1_small[1],p2_small[1],p3_small[1]))
        y_min = math.floor(min(p1_small[0],p2_small[0],p3_small[0]))
        #x_max = math.ceil(max(p1_small[0],p2_small[0],p3_small[0]))
        x_max = math.ceil(max(p1_small[1],p2_small[1],p3_small[1]))
        #y_max = math.ceil(max(p1_small[1],p2_small[1],p3_small[1]))
        y_max = math.ceil(max(p1_small[0],p2_small[0],p3_small[0]))
        
        # if (indice_triangle == 2):
        #     print(p1,p2,p3)
        #     print(p1_small,p2_small,p3_small)

        #for i in range(x_min,x_max):
        for i in range(y_min,y_max):
            for j in range(x_min,x_max):
                if (already_visited[i][j] == False):
                    if (isintriangle(p1_small, p2_small, p3_small, [i,j])):
                        cond_img[i][j] = delta_perm[indice_triangle]
                        already_visited[i][j] = True

        
    end2 = time.time()
    # print("Finding triangles: ",end2 - start2)
                    
                
    
        
                
    #parcourir les triangles puis pop les pixels            
                
                
    # if test:
    #     fig, ax = plt.subplots(1,2, constrained_layout=True)
    #     fig.set_size_inches(6, 4)
        
    #     im=ax[0].tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
    #     ax[0].set_aspect("equal")
    #     ax[0].set_title('image from triangulation')
    #     fig.colorbar(im, ax=ax[0])
        
    #     im_arr = ax[1].imshow(cond_img)
    #     ax[1].set_title('conductivity map')
    #     fig.colorbar(im_arr, ax=ax[1])
    #     plt.gca().invert_yaxis()
    #     plt.show()
                
    return cond_img

def label_data_rand(num, nb_px = 128, test=False):
    """"
    Compute and save the conductivity map and the voltage at the borders
    For one anomaly that is randomly generated
    
    Params:
    -----------
    num: int used for the name of the file
    nb_px: int 
    
    """
    
    #set the parameters of the random anomaly
    ray = (0.7-0.05)*rd.random() + 0.05 #RAY randomly between 0.04 and 0.7 
    #perm = (60000- 500)*rd.random() + 500
    #generate delta_perm between 0 and 1
    perm = rd.random()+1

    
    #conductivity is equal to 1 every where
    mesh_obj, el_pos = mesh.create(16, h0=0.1, fd=thorax)
    # extract node, element, alpha
    pts = mesh_obj["node"]  # array of coordinates (x,y) for each mesh's point 
    #For each triangle, the indices of the three points that make up the triangle, ordered in an anticlockwise manner.
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]
    
    ano_x, ano_y = (np.max(x)-np.min(x))*rd.random() + np.min(x), (np.max(y) - np.min(y))*rd.random()+ np.min(y)

    #add one random anomaly 
    anomaly = [{"x": ano_x, "y": ano_y, "d": ray, "perm": perm}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)
    
    delta_perm = mesh_new["perm"] - mesh_obj["perm"] #match each triangle to a conductivity value
    
    """ 1. Save the array containing the conductivity """
    cond_img = generate_array(nb_px, x, y, tri, delta_perm, test)
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
    
def label_data_sym(num, data='train', ray=0.1, perm=1000.0, test=False, nb_px = 128):
    """
    Labelization of data 
    --------------------
    Parameters: 
        num: int, numerotation of the generated data, it is used in the file names 
        ray: float, the ray of the simulated anomaly 
        
    """
    #conductivity is equal to 1 every where
    mesh_obj, el_pos = mesh.create(16, h0=0.1, fd=thorax)

    # extract node, element, alpha
    pts = mesh_obj["node"]  # Tableau de points qui constituent la discretisation du domaine
    #For each triangle, the indices of the three points that make up the triangle, ordered in an anticlockwise manner.
    tri = mesh_obj["element"] # Tableau de triangles. Un triangle = 1 tableau qui contient 3 indices qui identifient les points
    x, y = pts[:, 0], pts[:, 1]
    
    #add almost-symmetrical anomaly
    anomaly = [{"x": -0.25, "y": 0.0, "d": ray, "perm": perm},{"x": 0.5, "y": 0.0, "d": ray, "perm": perm}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)
    delta_perm = mesh_new["perm"] - mesh_obj["perm"] 
    
    """ 1. Save the array containing the conductivity """
    cond_img = generate_array(nb_px, x, y, tri, delta_perm, test=test)
    np.save('data_'+data+'/conductivity_images/class_images/'+str(num)+'_cond_img', cond_img)
    
    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    # adjacent stimulation (el_dist=1), adjacent measures (step=1)
    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(16, el_dist)
    
    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    np.save('data_'+data+'voltage_borders/class_images/'+str(num)+'_voltage_border',f1.v - f0.v)

def test(num=0):
    # start = time.time()
    label_data_rand(num, test=True)
    # end = time.time()
    # print("Execution time: ", end-start)
    #label_data_sym(num, test=True)
    
def generation_data(nb_data):
    for indice_generated_data in range(nb_data):
        if(indice_generated_data%100 == 0):
            print(indice_generated_data)
        test(indice_generated_data)
    
    
