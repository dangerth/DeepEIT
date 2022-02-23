"""create the EIT images and data using PyEIT"""
import json
import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.utils import eit_scan_lines
from utils import new_anomaly, EIT_to_image

import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
	        if isinstance(obj, np.ndarray):
	            return obj.tolist()
	        return json.JSONEncoder.default(self, obj)
	

	



n_phantom=3000
anomaly_type="healthylungs" #"damagedlungs" "healthylungs" "circles"
filename="phantoms_"+str(n_phantom)+"_"+anomaly_type+".json"#"phantoms_5k_circles"
"""clear file if it exists"""
with open(filename,"a+") as file:
	file.truncate(0)
	file.close()
plot_phantoms=False #will plot each created phantom for verification

img_size=128 #EIT phantoms will be saved as (img_size x img_size) images
background=False  #True will add "body" as background with conductivity=1



"""construct mesh, set up EIT parameters"""
"""see PyEIT documentation for details """
""" do not change settings as model dimensions might no longer fit """
with open("settings.json") as file: 
	settings=json.load(file)
	n_el=settings["n_el"]
	h0=settings["h0"]
	el_dist=settings["el_dist"]
	step=settings["step"]
	img_size=settings["img_size"]

mesh_obj, el_pos = mesh.create(n_el, h0=h0)
pts = mesh_obj["node"]
tri =mesh_obj["element"]
x, y = pts[:, 0], pts[:, 1]
ex_mat = eit_scan_lines(n_el, el_dist)

imgtrain=np.zeros((n_phantom,img_size,img_size,1))
vtrain=np.zeros((n_phantom,208)) ##calculate from ex_mat?

""" calculate centroids of triangles """
tricent=np.zeros((tri.shape[0],2))
for i in range(tri.shape[0]):
	coords=pts[tri[i]]
	cent=np.sum(coords,axis=0)/3
	tricent[i]=cent
	


for i in range(n_phantom):
#set up phantoms, calculate EIT data, convert phantom to image, save data and image
#image conversion: each pixel is assigned the conductivity of the triangle with centroid closest to pixel center coordinate"""
#this is much faster than finding the triange each pixel center lies in """
	if i%100==0:
		print("phantoms done:",i)
	anomalist=new_anomaly(anomaly_type)
	mesh_new = mesh.set_perm(mesh_obj, anomaly=anomalist)
	
	data=EIT_to_image(mesh_obj,el_pos,ex_mat,step,tricent,mesh_new["perm"],img_size,background,plot_phantoms)
	imgtrain[i]=data["img"]
	vtrain[i]=data["v"]



savedat={"img":imgtrain,"v":vtrain}
with open(filename,"a+") as file:
		json.dump(savedat,file,cls=NumpyEncoder)
		file.write('\n')
		file.close()


