from argparse import ArgumentParser
import airsimdroneracinglab as airsim
import threading
import time
import utils
import numpy as np
import math
from controller import simulate
import transformations
# import rospy
# from std_msgs.msg import Float64MultiArray
import cv2
import random
import tensorflow.compat.v1 as tf
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from scipy.spatial.transform import Rotation as R
import json 
import matplotlib.pyplot as plt 
import pyvista as pv 

MODEL_NAME = './object_detection'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, './object_detection', 'labelmap.pbtxt')

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #sess = tf.Session(graph=detection_graph,config=tf.ConfigProto(gpu_options=gpu_options))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=detection_graph,config=config)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def inference(img_rgb):
    frame_expanded = np.expand_dims(img_rgb, axis=0)
    (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
    
    index = np.squeeze(scores >= 0.90)
    boxes_detected = np.squeeze(boxes)[index]   # only interested in the bounding boxes that show detection
    
    N = len(boxes_detected)
    H_list = []
    W_list = []
    H, W, My, Mx = None, None, None, None
    detect_flag = False 
    if N >= 1:  # in the case of more than one gates are detected, we want to select the nearest gate (biggest bounding box)
        for element in boxes_detected:
            H_list.append(element[2] - element[0])
            W_list.append(element[3] - element[1])
        if N > 1:
            # print('boxes_detected', boxes_detected, boxes_detected.shape)
            Area = np.array(H_list) * np.array(W_list)
            max_Area = np.max(Area)
            idx_max = np.where(Area == max_Area)[0][0]  # find where the maximum area is
            # print(Area)
        else:
            idx_max = 0
        box_of_interest = boxes_detected[idx_max]
        h_box = box_of_interest[2]-box_of_interest[0]
        w_box = box_of_interest[3]-box_of_interest[1]
        Area_box = h_box * w_box
        # if N > 1:
        #     print('box_of_interest', box_of_interest, box_of_interest.shape)
        #     print('----------------------------------')
        if Area_box <= 0.98 and Area_box >= 0.01:    # Feel free to change this number, set to 0 if don't want this effect
            # If we detect the box but it's still to far keep the same control command
            # This is to prevent the drone to track the next gate when it has not pass the current gate yet
            detect_flag = True
            H = box_of_interest[2]-box_of_interest[0]
            W = box_of_interest[3]-box_of_interest[1]
            My = (box_of_interest[2]+box_of_interest[0])/2
            Mx = (box_of_interest[3]+box_of_interest[1])/2
            #print("boxes_detected : ", boxes_detected, "W : ", W, "H", H, "M : ", Mx, " ", My)
            # print("Area_box", Area_box)
            vis_util.visualize_boxes_and_labels_on_image_array(
                img_rgb,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.90
            )
        else:
            detect_flag = False
        #     print("=============== NOT DETECT ===============")
    else:
        # print('==================== set detect_flag to FALSE ====================')
        estimate_depth = 8
        detect_flag = False
    return detect_flag, H, W, My, Mx

def State(phase, idx=None, coord=None):
    return {'phase': phase, 'id': idx, 'coord': coord}

def to_list(x):
    return [x.x_val, x.y_val, x.z_val]

def distance(x, y):
    e = (np.array(x) - np.array(y))
    return np.sqrt((e ** 2).sum())

# drone_name should match the name in ~/Document/AirSim/settings.json
class BaselineRacer(object):
    def __init__(
        self,
        drone_name="drone_1",
        viz_traj=True,
        viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0],
        viz_image_cv2=True,
    ):
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.race_tier = 1
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03)
        )
        self.odometry_callback_thread = threading.Thread(
            target=self.repeat_timer_odometry_callback,
            args=(self.odometry_callback, 0.02),
        )
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = (
            10  # see https://github.com/microsoft/AirSim-Drone-Racing-Lab/issues/38
        )

    # loads desired level
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=3):
        self.airsim_client.simStartRace(tier)

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        self.airsim_client.simResetRace()

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )

        self.airsim_client.setTrajectoryTrackerGains(
            traj_tracker_gains, vehicle_name=self.drone_name
        )
        time.sleep(0.2)

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        takeoff_waypoint = airsim.Vector3r(
            start_position.x_val,
            start_position.y_val,
            start_position.z_val - takeoff_height,
        )

        self.airsim_client.moveOnSplineAsync(
            [takeoff_waypoint],
            vel_max=15.0,
            acc_max=5.0,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        ).join()

    # stores gate ground truth poses as a list of airsim.Pose() objects in self.gate_poses_ground_truth
    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        self.gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (
                math.isnan(curr_pose.position.x_val)
                or math.isnan(curr_pose.position.y_val)
                or math.isnan(curr_pose.position.z_val)
            ) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(
                curr_pose.position.x_val
            ), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.y_val
            ), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.z_val
            ), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)

    # this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints()
    # the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale=1.0):
        import numpy as np

        # convert gate quaternion to rotation matrix.
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array(
            [
                airsim_quat.w_val,
                airsim_quat.x_val,
                airsim_quat.y_val,
                airsim_quat.z_val,
            ],
            dtype=np.float64,
        )
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsim.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )
        gate_facing_vector = rotation_matrix[:3, 1]
        gate_facing_vector = gate_facing_vector / (np.sqrt((gate_facing_vector**2).sum()))
        return airsim.Vector3r(
            scale * gate_facing_vector[0],
            scale * gate_facing_vector[1],
            scale * gate_facing_vector[2],
        )

    def fly_through_all_gates_one_by_one_with_moveOnSpline(self):
        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0

        if self.level_name in [
            "Soccer_Field_Medium",
            "Soccer_Field_Easy",
            "ZhangJiaJie_Medium",
        ]:
            vel_max = 10.0
            acc_max = 5.0

        return self.airsim_client.moveOnSplineAsync(
            [gate_pose.position],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_at_once_with_moveOnSpline(self):
        if self.level_name in [
            "Soccer_Field_Medium",
            "Soccer_Field_Easy",
            "ZhangJiaJie_Medium",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ]:
            vel_max = 30.0
            acc_max = 15.0

        if self.level_name == "Building99_Hard":
            vel_max = 4.0
            acc_max = 1.0

        return self.airsim_client.moveOnSplineAsync(
            [gate_pose.position for gate_pose in self.gate_poses_ground_truth],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints(self):
        add_velocity_constraint = True
        add_acceleration_constraint = False

        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy"]:
            vel_max = 15.0
            acc_max = 3.0
            speed_through_gate = 2.5

        if self.level_name == "ZhangJiaJie_Medium":
            vel_max = 10.0
            acc_max = 3.0
            speed_through_gate = 1.0

        if self.level_name == "Building99_Hard":
            vel_max = 2.0
            acc_max = 0.5
            speed_through_gate = 0.5
            add_velocity_constraint = False

        # scale param scales the gate facing vector by desired speed.
        return self.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position],
            [
                self.get_gate_facing_vector_from_quaternion(
                    gate_pose.orientation, scale=speed_through_gate
                )
            ],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=add_velocity_constraint,
            add_acceleration_constraint=add_acceleration_constraint,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def fly_through_all_gates_at_once_with_moveOnSplineVelConstraints(self):
        if self.level_name in [
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
        ]:
            vel_max = 15.0
            acc_max = 7.5
            speed_through_gate = 2.5

        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0
            speed_through_gate = 1.0

        return self.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position for gate_pose in self.gate_poses_ground_truth],
            [
                self.get_gate_facing_vector_from_quaternion(
                    gate_pose.orientation, scale=speed_through_gate
                )
                for gate_pose in self.gate_poses_ground_truth
            ],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=True,
            add_acceleration_constraint=False,
            viz_traj=self.viz_traj,
            viz_traj_color_rgba=self.viz_traj_color_rgba,
            vehicle_name=self.drone_name,
        )

    def run(self, rect, real_gate_pos, num_sample = 1000):
        self.airsim_client.simEnableWeather(True)
        self.airsim_client.simSetWeatherParameter(0, 0.5)
        # self.airsim_client.simSetTimeOfDay(is_enabled=True)
        # res_list = []
        # for i in range(num_sample):
        #     print(f">>>>>>image{i}")
        #     # Sample drone position
        #     x = random.uniform(rect[0][0], rect[1][0])
        #     y = random.uniform(rect[0][1], rect[1][1])
        #     z = random.uniform(rect[0][2], rect[1][2])

        #     # Set drone position
        #     pose = self.airsim_client.simGetVehiclePose(
        #         vehicle_name=self.drone_name
        #     )
        #     pose.position.x_val = x 
        #     pose.position.y_val = y 
        #     pose.position.z_val = z 
        #     q = transformations.quaternion_from_euler(0, 0, -1.371073)
        #     pose.orientation.w_val = q[0]
        #     pose.orientation.x_val = q[1]
        #     pose.orientation.y_val = q[2]
        #     pose.orientation.z_val = q[3]
        #     self.airsim_client.simSetVehiclePose(pose=pose, vehicle_name = self.drone_name, ignore_collison=True)
            
        #     # Capture image 
        #     request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        #     response = self.airsim_client_images.simGetImages(request)
        #     img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        #     img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        #     cv2.imshow('Object detector', img_rgb)
        #     cv2.waitKey(1)
        #     cv2.imwrite(f"./picture_sc/data/{i}.png", img_rgb)
            
        #     # Perform perception 
        #     detect_flag, H, W, My, Mx = inference(img_rgb)
            
        #     # Compute gate position 
        #     estimated_gate_pos = []
        #     if detect_flag:
        #         # Estimated distance between drone and gate 
        #         K1 = 25.2576331
        #         K2 = -4.09327744

        #         # K1 = 29.94319383
        #         # K2 = -5.64534919                
        #         d_hat = K1*np.exp(K2*W)

        #         # K1 = 40.94948766
        #         # K2 = -9.64960739
        #         # K3 = 2.35191126
        #         # d_hat = K1*np.exp(K2*W)+K3

        #         # Estimated yaw error between drone and gate 
        #         ey = (20 * (Mx - 0.5))
        #         e_yaw = np.arctan2(ey, d_hat)
        #         # max_e_yaw = max(max_e_yaw, abs(e_yaw))

        #         # Estimated pitch error between drone and gate
        #         ez = (20 * (My - 0.5))
        #         e_pitch = np.arctan2(ez, d_hat)

        #         drone_x = x
        #         drone_y = y
        #         drone_z = z

        #         roll_drone = 0
        #         pitch_drone = 0
        #         yaw_drone = -1.371073

        #         # r_drone = R.from_quat([qx,qy,qz,qw])
        #         # roll_gate = roll_drone
        #         # pitch_gate = pitch_drone+e_pitch
        #         # yaw_gate = yaw_drone+e_yaw
        #         # r_gate = R.from_euler('xyz',[roll_gate, pitch_gate, yaw_gate])
        #         # estimated_gate_pos = r_gate.apply([d_hat,0,0]) + np.array([drone_x, drone_y, drone_z])
        #         r = R.from_euler('xyz',[roll_drone, pitch_drone, yaw_drone])
        #         gate_pos_drone = [d_hat, ey, ez]
        #         # tmp = np.linalg.inv(r.as_matrix())@gate_pos_drone*np.array([-1,1,-1]) 
        #         tmp = r.apply(gate_pos_drone)
        #         estimated_gate_pos = tmp + np.array([drone_x, drone_y, drone_z])
        #         # print([roll_drone, pitch_drone, yaw_drone])
        #         # print(tmp)
        #         # print(Mx, My, d_hat, ey, ez)
        #         print(estimated_gate_pos)
        #     else:
        #         print("gate not detected")            

        #     # Store results
        #     s_star = (np.array(real_gate_pos) - np.array([x,y,z])).tolist()
        #     if detect_flag:
        #         s = (np.array(estimated_gate_pos) - np.array([x,y,z])).tolist()
        #     else:
        #         s = []

        #     if W is None:
        #         W = -1000
        #     res = {
        #         "s_star": s_star,
        #         "s": s,
        #         "xs": [x,y,z],
        #         "a_star": real_gate_pos,
        #         "a": list(estimated_gate_pos),
        #         "detect_flag": detect_flag,
        #         'W':float(W)
        #     }
        #     with open(f"./picture_sc/data/{i}.json","w+") as f:
        #         json.dump(res, f)
        #     cv2.imwrite(f"./picture_sc/data/{i}_detect.png", img_rgb)
        #     # time.sleep(0.001)
        #     res_list.append(res)
        # return res_list 
        return []

    def next_state(self, curr_pose, state):
        if state['phase'] == 'init':
            state = State('before', 0, self.before_gate(0))
        else:
            threshold = 0.1
            if distance(np.array(curr_pose)[[0,4,8]], state['coord']) < threshold:
                if state['phase'] == 'before':
                    state['phase'] = 'center'
                    state['coord'] = self.center_gate(state['id'])
                elif state['phase'] == 'center':
                    state['phase'] = 'after'
                    state['coord'] = self.after_gate(state['id'])
                elif state['phase'] == 'after':
                    state['phase'] = 'before'
                    if state['id'] == len(self.gate_poses_ground_truth) - 1:
                        state['phase'] = 'terminate'
                    else:
                        state['id'] += 1
                        state['coord'] = self.before_gate(state['id'])
                else:
                    raise ValueError('wrong state')
        return state

    def before_gate(self, idx):
        gate_pose = self.gate_poses_ground_truth[idx]
        return to_list(gate_pose.position - self.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale=1.0))

    def center_gate(self, idx):
        gate_pose = self.gate_poses_ground_truth[idx]
        return to_list(gate_pose.position)

    def after_gate(self, idx):
        gate_pose = self.gate_poses_ground_truth[idx]
        return to_list(gate_pose.position + self.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale=1.0))

    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if self.viz_image_cv2:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)

    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        position = drone_state.kinematics_estimated.position
        orientation = drone_state.kinematics_estimated.orientation
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        angular_velocity = drone_state.kinematics_estimated.angular_velocity

    # call task() method every "period" seconds.
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")

def compute_accuracy(rm, dataset):
    num_accurate = 0
    total_points = 0
    max_rm = -float('inf')
    dist_list = []
    for data in dataset:
        detect_flag = data['detect_flag']
        if detect_flag:
            total_points += 1
            s = data['s']
            s_star = data['s_star']
            max_rm = max(max_rm, distance(s,s_star))
            dist = distance(s, s_star)
            dist_list.append(dist)
            if dist<rm:
                num_accurate += 1
    plt.hist(dist_list)
    plt.savefig('tmp.png')    
    accuracy = num_accurate / total_points
    print(max_rm, total_points, num_accurate, accuracy)
    return num_accurate, accuracy

def visualize(dataset, real_gate_pos, rect_center, rect_radius):
    plotter = pv.Plotter()
    estimated_pos = []
    for data in dataset:
        if data['detect_flag']:
            estimated_pos.append(data['a'])
    estimated_pos = np.array(estimated_pos)
    point_cloud = pv.PolyData(estimated_pos)
    plotter.add_mesh(point_cloud, color='b', point_size=10.,
                 render_points_as_spheres=True)
    real_gate_pos = np.array([real_gate_pos])
    point_cloud2 = pv.PolyData(real_gate_pos)
    plotter.add_mesh(point_cloud2, color='g', point_size=20.,
                 render_points_as_spheres=True)
    mesh = pv.Cube(
        center=tuple(rect_center), 
        x_length=rect_radius[0], 
        y_length=rect_radius[1], 
        z_length=rect_radius[2]
    )
    plotter.add_mesh(mesh, 'r', opacity = 0.5)
    plotter.show()

def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer = BaselineRacer(
        drone_name="drone_1",
        viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0],
    )
    baseline_racer.load_level(args.level_name)
    baseline_racer.get_ground_truth_gate_poses()

    # Specify parameters for performing random sampling
    # Specify the center and radius of the rectangle
    center = [-51.577632904052734, 118.41986083984375, 0.673719048500061]
    # center = [-51.330406188964844, 113.5843048095703, 0.2118362933397293]
    # center = [-51.577632904052734, 115.41986083984375, 0.673719048500061]
    radius = np.array([1.0, 1.0, 1.0])
    # radius = np.array([3, 8.5, 2.5])
    rect = [(np.array(center)-radius).tolist(), (np.array(center)+radius).tolist()]
    
    # Specify the real position of the gate
    gate_index = 6 
    real_gate_pos = [-49.43488693237305, 105.03890228271484, 0.8199999928474426]

    # Specify the number of samples we want
    num_sample = 1000

    # Randomly sample position and inference gate position
    res = baseline_racer.run(rect,real_gate_pos, num_sample)
    # Compute the inferencing accuracy
    num_accurate, accuracy = compute_accuracy(1, res)

    visualize(res, real_gate_pos, center, radius)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--level_name",
        type=str,
        choices=[
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
            "Building99_Hard",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ],
        default="Final_Tier_2",
    )
    args = parser.parse_args()
    main(args)
