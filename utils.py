import numpy as np
from pyeit.eit.fem import Forward
import matplotlib.pyplot as plt



""" calculates EIT forward solution, converts FE net to image """
def EIT_to_image(mesh_obj,el_pos,ex_mat,step,tricent,perm,img_size,background,plot_phantoms=False):
#set up phantoms, calculate EIT data, convert phantom to image, save data and image
#image conversion: each pixel is assigned the conductivity of the triangle with centroid closest to pixel center coordinate"""
#this is much faster than finding the triange each pixel center lies in """
	imgtrain=np.zeros((1,img_size,img_size,1))
	delta=2/(img_size-1)
	fwd = Forward(mesh_obj, el_pos)
	f = fwd.solve_eit(ex_mat, step=step, perm=perm)
	for ix in range(img_size):
		for iy in range(img_size):
			pt_x=-1+ix*delta
			pt_y=-1+iy*delta
			if pt_x**2+pt_y**2<=1:
				pt_test=[pt_x,pt_y]
				dists=np.sum(np.square(tricent-pt_test),axis=1)
				idx=np.argmin(dists,axis=0)
				imgtrain[0,img_size-1-iy,ix,0]=perm[idx] if background else perm[idx]-1

	if plot_phantoms:	
		fig=plt.figure()
		plt.imshow(np.squeeze(imgtrain))
		plt.show()
		
	savedat={"v":np.array(f.v),"img":imgtrain}
	return savedat

"""produces the phantoms used for training """
def new_anomaly(key):
	anomalist=[]
	if key=="healthylungs":
		for i in range(2):
			x_cent=np.random.normal((-1)**i*0.45,0.1)
			y_cent=np.random.normal(0,0.1)
			d_circ=0.3+0.1*np.random.random_sample()
			perm_val=2 
			anomalist.append({"x": x_cent, "y": y_cent, "d": d_circ, "perm": perm_val})
			
	elif key=="damagedlungs":
		for i in range(2):
			p_defect=np.random.uniform()
			if p_defect>0.15: #undamaged lung
				x_cent=np.random.normal((-1)**i*0.45,0.1)
				y_cent=np.random.normal(0,0.1)
				d_circ=0.35+0.1*np.random.random_sample()
				perm_val=2+np.random.normal(0,0.25)
				anomaly={"x": x_cent, "y": y_cent, "d": d_circ, "perm": perm_val,"defect":"none"}
			elif p_defect<0.02: #no lung
				anomaly={"x": 0, "y": 0, "d": 0, "perm": 1,"defect":"full"}
			else:#damaged lung
				x_cent=np.random.normal((-1)**i*0.45,0.2)
				y_cent=np.random.normal(0,0.2)
				d_circ=0.15+0.1*np.random.random_sample()
				perm_val=1.5+np.random.normal(0,0.5)
				anomaly={"x": x_cent, "y": y_cent, "d": d_circ, "perm": perm_val,"defect":"semi"}
			anomalist.append(anomaly)
	
	elif key=="circles":
		n_subobjects=np.random.randint(1,7);
		for i in range(n_subobjects):
			x_cent=np.random.normal(0,0.5)
			y_cent=np.random.normal(0,0.5)
			d_circ=0.5*np.random.random_sample()+0.01
			perm_val=2*np.random.random_sample()-1
			anomalist.append({"x": x_cent, "y": y_cent, "d": d_circ, "perm": perm_val})
		
	
	else:
		anomaly=0#dummy
			
	return anomalist
