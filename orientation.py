#========================================================#
#     Author: Imoleayo Abel                              #
#     Course: Sensing and Estimation Robotics (ECE 276A) #
#    Quarter: Fall 2017	                                 #
# Instructor: Nikolay Atanasov                           #
#    Project: 02 - Orientation Tracking                  #
#       File: orientation.py                             #
#       Date: Nov-22-2017                                #
#========================================================#
import numpy as np
import myquaternion as mq
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import os, pickle

# compute scale factors for gyroscope and accelerometer data
vref = 3300.0
accel_scl = vref/(1023*330)
gyro_scl = np.pi*vref/(1023*3.33*180)

# folder to read input from
folder = "testset"

# process each IMU file in <folder>/imu/
for file_name in os.listdir(folder + "/imu/"):	
	# only consider .p files
	if file_name.lower().endswith(".p"):
		file_num_str = file_name.lower().split("imuraw")[1].split('.')[0]

		# load IMU data
		imu_data_filename = folder + "/imu/imuRaw" + file_num_str + ".p"
		print "\n\nAttempting to load", imu_data_filename + "..."
		imu_data = pickle.load( open( imu_data_filename, "rb" ) )
		print "Done!"

		# get data dimensions
		imu_v, imu_t = imu_data['vals'], imu_data['ts'][0]
		imu_cnt = imu_v.shape[1]

		# scale and normalize data
		imu_v = imu_v - np.mean(imu_v[:, :100], axis=1).reshape((6,1)) # subtract bias 
		imu_v[:3, :] = accel_scl * imu_v[:3, :]		 # scale acceleration
		imu_v[:2, :] = -imu_v[:2, :]                 # negate ax and ay 
		imu_v[2, :] = imu_v[2,:] + 1                 # normalize so that az-component is 1 at start
		imu_v[3:, :] = gyro_scl * imu_v[[4,5,3], :]  # reorder and scale wx, wy, wz

		############################## 
		#     Simple Integration     #
		##############################
		print "Computing integration-only orientation..."

		# initialize array of Euler angles and initial orientation
		integration_eulers = np.zeros((imu_cnt,3))
		q = np.array([1,0,0,0])
		integration_eulers[0] = mq.quat2Euler(q)

		# integrate gyro data over time
		for i in range(1,imu_cnt):
			dt = imu_t[i] - imu_t[i-1]
			omega = imu_v[3:, i]
			q = mq.qMult(q, mq.qExp(np.concatenate(([0], 0.5*dt*omega))))
			integration_eulers[i] = mq.quat2Euler(mq.qUnitize(q))
		print "Done!"

		# setup plots
		f, axes = plt.subplots(2,2)
		f.canvas.set_window_title("Data set: #" + file_num_str)
		f.set_size_inches(12, 10, forward=True)

		#############################################
		#     Vicon Ground Truth (if it exists)     #
		#############################################
		vic_data_filename = folder + "/vicon/viconRot" + file_num_str + ".p"
		print "Attempting to load", vic_data_filename+"..."

		# load vicon data if it exists
		vic_data_exists = False
		vic_data = ""
		try:
			vic_data = pickle.load( open( vic_data_filename, "rb" ) )
			vic_data_exists = True
			print "Done!"
		except:
			print "Not found :("

		# convert vicon data to Euler angles and plot for comparison
		if vic_data_exists:
			vic_v, vic_t = vic_data['rots'], vic_data['ts'][0]
			vic_cnt = vic_v.shape[2]
			vic_eulers = np.zeros((vic_cnt,3))
			for i in range(vic_cnt):
				vic_eulers[i] = mq.rot2Euler(vic_v[:,:,i])
			axes[0,0].plot(vic_t,vic_eulers[:,0],'b',label="vicon")
			axes[0,1].plot(vic_t,vic_eulers[:,1],'b',label="vicon")
			axes[1,0].plot(vic_t,vic_eulers[:,2],'b',label="vicon")
		
		# plot integration-only data
		axes[0,0].plot(imu_t,integration_eulers[:,0],'y',label="integration")		
		axes[0,1].plot(imu_t,integration_eulers[:,1],'y',label="integration")		
		axes[1,0].plot(imu_t,integration_eulers[:,2],'y',label="integration")
		# caption subplots
		axes[0,0].set_title("Roll")
		axes[0,1].set_title("Pitch")
		axes[1,0].set_title("Yaw")

		######################################
		#     Unscented Kalman Filtering     #
		######################################
		print "Performing UKF..."

		# initialize required variables
		update_eulers = np.zeros((imu_cnt,3))         # array of Euler angles
		P_pred = 0.0001*np.eye(6)                     # state covariance matrix
		Q = 0.0000000000001*np.eye(6)                          # process noise covariance matrix
		R = 0.00065*np.eye(3)                         # measurement noise covariance matrix
		omega_pred = imu_v[3:,0]                      # initial gyro data
		q_pred = np.array([1, 0, 0, 0])               # initial orientation quaternion
		update_eulers[0]  = mq.quat2Euler(q_pred)     # save initial orientation Euler angles
		g_vec = np.array([0,0,1])                     # gravity vector
		sigma_Ws_prime = np.zeros((6,13))             # sigma points for prediction
		sigma_Zs = np.zeros((3,13))                   # sigma points for update

		# iterate over time
		for i in range(1,imu_cnt):
			S = np.sqrt(6) * np.linalg.cholesky(P_pred + Q)
			sigma_Ws = np.concatenate((S.T, -S.T), axis=1)

			# predict quaternion portion of predicted state and covariance
			dt = imu_t[i] - imu_t[i-1]
			q_delta = mq.qExp(np.concatenate(([0], 0.5*omega_pred*dt)))

			# apply process model to first sigma point (mean), and then project into measurement 
			# space using measurement model
			sigma_Y = mq.qMult(q_pred, q_delta)
			sigma_Ys = [sigma_Y]
			sigma_Zs[:,0] = mq.qRotate(mq.qInv(sigma_Y), g_vec)

			# apply process model to all non-mean sigma points, and then project into measurement 
			# space using measurement model
			for j in range(12):
				sigma_X = mq.qMult(q_pred, \
					             mq.qExp(np.concatenate(([0], 0.5*sigma_Ws[:3, j].reshape((3,))))))
				sigma_Y = mq.qMult(sigma_X, q_delta)
				sigma_Ys.append(sigma_Y)
				sigma_Zs[:,j+1] = mq.qRotate(mq.qInv(sigma_Y), g_vec)

			# average all post-process-model sigma points to get prediction of state mean
			mean_weigths = [0] + [1/12.0]*12
			q_pred = mq.qAvg(sigma_Ys, mean_weigths)

			# subtract mean, and then convert quaternion part of sigma points to rotation vector to
			# be used for computing prediction of state covariance
			for j in range(13):
				sigma_Ws_prime[:3,j] = (2*mq.qLog(mq.qMult(sigma_Ys[j], mq.qInv(q_pred))))[1:]

			# get angular velocity portion for state covariance prediction
			sigma_Ws_prime[3:,:] = np.concatenate((np.zeros((3,1)), sigma_Ws[3:, :]), axis=1)

			# compute (weighted) prediction covariance
			P_pred = (sigma_Ws_prime[:,1:].dot(sigma_Ws_prime[:,1:].T))/12.0 + \
			         2*(sigma_Ws_prime[:,0].reshape((6,1)).dot(sigma_Ws_prime[:,0].reshape((1,6))))

			# compute mean of measurement esitmate (Note: mean of 2nd-to-13th column is equivalent
			# to weighted sum of columns with column 1 weight = 0, and column 2-13 weights = 1/12)
			z_pred = np.mean(sigma_Zs[:,1:], axis=1) 

			# compute (weighted) covariance of measurement estimate
			sigma_Zs_offset = sigma_Zs - z_pred.reshape((3,1))
			P_zz = (sigma_Zs_offset[:, 1:].dot(sigma_Zs_offset[:, 1:].T))/12.0 + \
			       2*(sigma_Zs_offset[:,0].reshape((3,1)).dot(sigma_Zs_offset[:,0].reshape((1,3))))

			# compute innovation (actual measurement minus measurement estimate)
			innov = imu_v[:3,i] - z_pred.reshape((3,))

			# innovation covariance
			P_vv = P_zz + R

			# compute (weighted) cross-correlation between state noise and measurement noise
			P_xz = (sigma_Ws_prime[:,1:].dot(sigma_Zs_offset[:,1:].T))/12.0 + \
			       2*(sigma_Ws_prime[:,0].reshape((6,1)).dot(sigma_Zs_offset[:,0].reshape((1,3))))

			# kalman gain
			K = P_xz.dot(np.linalg.inv(P_vv))

			# update predicted state mean and covariance
			q_pred = mq.qMult(q_pred, mq.qExp(np.concatenate(([0], 0.5*K.dot(innov)[:3]))))
			omega_pred = imu_v[3:,i]
			P_pred = P_pred - K.dot(P_vv.dot(K.T))

			# save Euler angles corresponding to state mean
			update_eulers[i] = mq.quat2Euler(q_pred)
			
		print "Done!"

		# plot UKF orientation estimates
		axes[0,0].plot(imu_t,update_eulers[:,0],'r',label="update")
		axes[0,1].plot(imu_t,update_eulers[:,1],'r',label="update")
		axes[1,0].plot(imu_t,update_eulers[:,2],'r',label="update")
		# set axes labels and show legend
		for (ax_x, ax_y) in [(0,0), (0,1), (1,0)]:
			axes[ax_x, ax_y].legend(shadow=True)
			axes[ax_x, ax_y].set_xlabel('time')
			axes[ax_x, ax_y].set_ylabel("angle (rad)")

		########################################################
		#      Panorama Generation (if camera data exists)     #
		########################################################
		cam_data_filename = folder + "/cam/cam" + file_num_str + ".p"
		print "Attempting to load", cam_data_filename+"..."

		# load camera data
		cam_data_exists = False
		cam_data = ""
		try:
			cam_data = pickle.load( open( cam_data_filename, "rb" ) )
			cam_data_exists = True
			print "Done!"
		except:
			print "Not found :("

		# create panorama
		if cam_data_exists:
			# get cam data dimension
			cam_v, cam_t = cam_data['cam'], cam_data['ts'][0]
			n_rows, n_cols, n_chanls, cam_cnt = cam_v.shape

			# initialize panorama image
			pan_rows, pan_cols = n_rows, 3*n_rows/2
			panorama = np.zeros((pan_rows, pan_cols, 3), dtype=cam_v.dtype)

			# compute Cartesian coordinates of pixels in camera frame from polar coordinates
			cam_cartesian = np.ones((n_rows, n_cols, n_chanls))
			min_angle, angle_range = mq.deg2Rad(67.5), mq.deg2Rad(45) 
			for i in range(n_rows):
				theta = min_angle + i*angle_range/(n_rows-1)
				sin_t = np.sin(theta)
				cam_cartesian[i,:,:] = [sin_t, sin_t, np.cos(theta)]
			min_angle, angle_range = mq.deg2Rad(-30), mq.deg2Rad(60)
			for i in range(n_cols):
				phi = min_angle + i*angle_range/(n_cols-1)
				cam_cartesian[:,i,0] = np.cos(phi) * cam_cartesian[:,i,0]
				cam_cartesian[:,i,1] = np.sin(phi) * cam_cartesian[:,i,1]

			# reshape to improve performance
			cam_cartesian = cam_cartesian.reshape((n_rows*n_cols, n_chanls)).T

			# loop over each image in cam data
			print "Creating panorama..."
			imu_t_index = 0
			for i in range(cam_cnt):
				# use camera timestamp to find closest-in-the-past timestamp for filtered IMU data
				cam_time = cam_t[i]
				while (imu_t_index + 1 < imu_cnt) and (imu_t[imu_t_index + 1] <= cam_time):
					imu_t_index += 1

				# convert Cartesian coordinate of pixels fron camera frame to world frame
				R_c2w = mq.euler2Rot(update_eulers[imu_t_index])
				world_coord = R_c2w.dot(cam_cartesian) + np.array([0,0,0.1]).reshape((3,1))

				# normalize world Cartesian coordinates to unit norm to fit on unit sphere
				world_coord = world_coord / np.linalg.norm(world_coord, axis=0) 
				
				# convert world Cartesian coordinates to cylinderical coordinates and unwrap
				# (shift/scale) to fill dimension of panorama
				world_coord[0,:] = (0.5*(pan_cols - 1)/np.pi) * (np.pi + \
				                        np.arctan2(world_coord[1,:], world_coord[0,:])) # azimuth
				world_coord[1,:] = ((pan_rows - 1)/np.pi) * \
				                                        np.arccos(world_coord[2,:]) # inclination
				
				# copy pixels to appropriate location in panorama
				panorama[np.round(world_coord[1,:]).astype(np.int32), \
				         np.round(world_coord[0,:]).astype(np.int32)] = \
					                              cam_v[:,:,:,i].reshape((n_rows*n_cols, n_chanls))
			print "Done!"

			# display panorama
			axes[1,1].imshow(panorama)
			axes[1,1].set_title("Panorama")

		# display plot
		plt.show()
