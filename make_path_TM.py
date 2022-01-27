import numpy as np
import pyproj
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interpolate
import pandas as pd
from numpy.linalg import inv
############for rad to degree####################
PI = 3.141592
############for kalman filter####################
A = np.array([[ 1, 1,  0,  0],
              [ 0,  1,  0,  0],
              [ 0,  0,  1, 1],
              [ 0,  0,  0,  1]])
H = np.array([[ 1,  0,  0,  0],
              [ 0,  0,  1,  0]])
Q = 1.0 * np.eye(4)
R = np.array([[100,  0],
              [ 0, 100]])
############for find center point################
center = pd.read_csv('./cut_origin_TM.csv')
center_list = []
interpolate_list = []
this_pose = None
to_center = None
waypoint = []
new_waypoint = []
gps_data = []
####get waypoint####
def filter_data():
	gps_file = pd.read_csv('./GPS_Template1.csv')
	Lat = list(gps_file['Lat(rad)'].values)
	Long = list(gps_file['Long(rad)'].values)
	
	for la,lo in zip(Lat,Long):
		gps_data.append([la,lo])
	return gps_data

####rad to degree####
def to_degree_and_to_tm(x,y):
	x2 = x*180/PI
	y2 = y*180/PI
	
	LATLONG_WGS84 = pyproj.Proj("+proj=latlong +datum=WGS84 +ellps=WGS84")
	TM127 = pyproj.Proj("+proj=tmerc +lat_0=38N +lon_0=127E +ellps=bessel +x_0=200000 +y_0=600000 +k=1.0 ")
	longt = (float)(x2)
	lat = (float)(y2)
	x,y = pyproj.transform(LATLONG_WGS84, TM127, longt, lat)
	return x,y
	
####interpolate####
def interpolate_b_spline_path(x: list, y: list, n_path_points: int,
                              degree: int = 3) -> tuple:

    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)

    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)
    
####kalman_filter####
def kalman_filter(z_meas, x_esti, P):
	x_pred = A @ x_esti
	P_pred = A @ P @ A.T + Q 
	K = P_pred @ H.T @ inv(H @ P_pred @ H.T +R)
	x_esti = x_pred + K @ (z_meas - H @ x_pred)
	P = P_pred - K @ H @ P_pred
	return x_esti, P

def kalman_gps(spline_list):
	x_0 = np.array([spline_list[0][0], 0, spline_list[0][1], 0])  # (x-pos, x-vel, y-pos, y-vel) by definition in book.
	P_0 = 100 * np.eye(4)
	n_samples = len(spline_list)
	xpos_meas_save = np.zeros(n_samples)
	ypos_meas_save = np.zeros(n_samples)
	xpos_esti_save = np.zeros(n_samples)
	ypos_esti_save = np.zeros(n_samples)

	x_esti, P = None, None
	for i in range(n_samples):
		pos = spline_list
		z_meas = pos[i]
		if i == 0:
			x_esti, P = x_0, P_0
		else:
			x_esti, P = kalman_filter(z_meas, x_esti, P)
		xpos_meas_save[i] = z_meas[0]
		ypos_meas_save[i] = z_meas[1]
		xpos_esti_save[i] = x_esti[0]
		ypos_esti_save[i] = x_esti[2]
	kalman_data = []
	for k_x,k_y in zip(xpos_esti_save,ypos_esti_save):
		kalman_data.append([k_x,k_y])
	return kalman_data	
	
####make [x,x2],[y,y2] to list[[x,y],[x2,y2]]
def get_list(xlist,ylist):
	get_list = []
	for x,y in zip(xlist,ylist):
		data = [x,y]
		get_list.append(data)
	return get_list

####main####
def main():
	fordata = filter_data()
	deg_list = []
	data_x = []
	data_y = []
	this_distance = 10
    ####rad to degree####
	for i in fordata:
		todeg_x,todeg_y = to_degree_and_to_tm(i[1],i[0])
		todegree = [todeg_x,todeg_y]
		deg_list.append(todegree)
	for i in deg_list:
		data_x.append(i[0])
		data_y.append(i[1])
	####bspline interpolate####
	n_course_point = 500
	rix, riy = interpolate_b_spline_path(data_x, data_y,
                                         n_course_point)
	interpolate_list = get_list(rix,riy)
	final = pd.DataFrame(interpolate_list)
	#final.to_csv("./fix_interpolate_500_kalman.csv")
	after_kalman_list = kalman_gps(interpolate_list)
	####find close center point algorithm####
	for a,b in zip(center['x'],center['y']):
		center_list.append([a,b])
	before_center = [0,0]
	for this in after_kalman_list:
		this_pose = this
		for center_point in center_list:
			distance = np.hypot(this_pose[0]-center_point[0],this_pose[1]-center_point[1])
			if this_distance > distance:
				this_distance = distance
				to_center = center_point
#		print(np.hypot(before_center[0]-to_center[0],before_center[1]-to_center[1]))
		if np.hypot(before_center[0]-to_center[0],before_center[1]-to_center[1])>4.5:
			print(np.hypot(before_center[0]-to_center[0],before_center[1]-to_center[1]))
			pass 
		else:
			waypoint.append(to_center)
		before_center = to_center
		this_distance = 10
	'''		
	print(waypoint[0][0])
	for i in waypoint:
		w = i-np.array([0.00001,-0.0001])
		new_waypoint.append(w)
	print(new_waypoint[0][0])
	'''
	link_dataframe = pd.DataFrame(waypoint)
	link_dataframe.to_csv("./scenario1_fixed4.5.csv")	

if __name__ == '__main__':
    main()
    #filter_data()
