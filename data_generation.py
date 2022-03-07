# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:58:13 2022

@author: INES
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.mesh.shape import thorax
import matplotlib.pyplot as plt

""" 0. Construction du maillage """
# Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
mesh_obj, el_pos = mesh.create(16, h0=0.1, fd=thorax) # La conductivite est egale a 1 partout, par defaut
# mesh_obj, el_pos = mesh.layer_circle()

# extract node, element, alpha
pts = mesh_obj["node"]  # Tableau de points qui constituent la discretisation du domaine
tri = mesh_obj["element"] # Tableau de triangles. Un triangle = 1 tableau qui contient 3 indices qui identifient les points
x, y = pts[:, 0], pts[:, 1]

""" 1. Ajout d'une anomalie de conductivite egale a 1000 a point 0.5,0.5 avec un rayon de 0.1 """
anomaly = [{"x": 0.5, "y": 0.5, "d": 0.1, "perm": 1000.0}]
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

fig, axes = plt.subplots(1, constrained_layout=True)
fig.set_size_inches(6, 4)

ax = axes
delta_perm = mesh_new["perm"] - mesh_obj["perm"]
im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
ax.set_aspect("equal")
plt.show()

""" 2. FEM forward simulations """