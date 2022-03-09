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

# reset the plot configurations to default
#plt.rcdefaults()
# set the axes color glbally for all plots
#plt.rcParams.update({'axes.facecolor':'black'})

nb_px = 128

def label_data(nb, ray=0.1, perm=1000.0):
    """
    Labelization of data 
    --------------------
    Parameters: 
        nb: int, numerotation of the generated data, it is used in the file names 
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
    anomaly = [{"x": -0.25, "y": 0.0, "d": ray, "perm": perm}, {"x": 0.5, "y": 0.0, "d": ray, "perm": perm}]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)
    
    #save figure
    fig, ax = plt.subplots(1, constrained_layout=True)
    fig.set_size_inches(6, 4)
    
    
    delta_perm = mesh_new["perm"] - mesh_obj["perm"] #match each triangle to a conductivity value
    print("nb_pt: ", delta_perm.shape)
    print("nb_x: ", x.shape)
    print("nb_y: ", y.shape)
    print("nb_triangle: ", tri.shape)

    ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
    img = np.zeros((nb_px, nb_px))
    
    ax.set_aspect("equal")
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    # plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    # ax.set_facecolor("black")
    # plt.savefig('labeled_data/'+str(nb)+'_conductivity_img.jpg')
    # plt.close() #prevent the figure from being plotted
    plt.show()
    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    # adjacent stimulation (el_dist=1), adjacent measures (step=1)
    # el_dist, step = 1, 1
    # ex_mat = eit_scan_lines(16, el_dist)
    
    # # calculate simulated data
    # fwd = Forward(mesh_obj, el_pos)
    # f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    # f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])
    # #with open("my_file", "w") as f:

    # np.save('labeled_data/'+str(nb)+'_voltage_border',f1.v - f0.v)

#Test
#V= np.load('labeled_data/0_voltage_border.npy')
#print('TEST: ', V.shape)

#loop
def generation():
    
    ray = np.arange(0.015, 0.5, 0.005)
    perm = np.linspace(10, 50000, 1000)
    nb=1
    for r in ray:
        for p in perm:
            label_data(nb, ray=r, perm=p)
            nb += 1
            
    print('number of generated data: ', nb+1)
    


