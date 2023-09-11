from argparse import ArgumentParser
# from http.client import GATEWAY_TIMEOUT
import airsimdroneracinglab as airsim
import threading
import time
# import utils
import numpy as np
import math
# from controller import simulate
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
import copy

MODEL_NAME = './object_detection'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, './object_detection', 'labelmap.pbtxt')

NUM_CLASSES = 1

script_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_dir, './data/')
if not os.path.exists(data_path):
    os.makedirs(data_path)
img_path = os.path.join(script_dir, './img/')
if not os.path.exists(img_path):
    os.makedirs(img_path)

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
        self.gate_name_list = None
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

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

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

        self.airsim_client.race_tier = 1
        self.gate_poses_ground_truth = []
        self.gate_names_list = copy.deepcopy(gate_names_sorted)
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
            # print(gate_name)
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

    def apply_effect(self, image, effect_idx):
        if effect_idx == 0:
            image = image
        elif effect_idx == 1:
            # Blurr 
            ksize = (5, 5)
            image = cv2.blur(image, ksize)
        elif effect_idx == 2:
            # Increase contrast
            contrast = 192
            contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            image = cv2.addWeighted(image, Alpha,
                              image, 0, Gamma)
            pass  
        elif effect_idx == 3:
            # Decrease contrast
            contrast = 64
            contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            image = cv2.addWeighted(image, Alpha,
                              image, 0, Gamma)
            pass 
        elif effect_idx == 4:
            # Increase brightness 
            brightness = 400
            brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
            if brightness > 0: 
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
    
            al_pha = (max - shadow) / 255
            ga_mma = shadow
            # The function addWeighted calculates
            # the weighted sum of two arrays
            image = cv2.addWeighted(image, al_pha,
                                image, 0, ga_mma)
            pass 
        elif effect_idx == 5:
            # Decrease brightness 
            brightness = 64
            brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
            if brightness > 0: 
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
    
            al_pha = (max - shadow) / 255
            ga_mma = shadow
            # The function addWeighted calculates
            # the weighted sum of two arrays
            image = cv2.addWeighted(image, al_pha,
                                image, 0, ga_mma)
            pass 
        elif effect_idx == 6:
            # Add salt&papper noise
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            image = out
            pass 
        elif effect_idx == 7:
            # Add gaussian noise
            row,col,ch= image.shape
            mean = 0
            var = 0.5
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            image = noisy
        else:
            image = image 
        return image 

    def sample_drone_pose(self, gate_pos:np.ndarray, gate_ori: R, gate_dist: float):
        max_dist = gate_dist - 3.5-1
        drone_pos_dist = np.random.uniform(max_dist)
        if drone_pos_dist > 30:
            drone_pos_dist = 30
        drone_pos_ori = np.random.uniform(-np.arctan2(10, 13.5),np.arctan2(10, 13.5))
        drone_height = np.random.uniform(-4/(17-3.5)*drone_pos_dist, 4/(17-3.5)*drone_pos_dist)
        drone_yaw = np.random.uniform(-drone_pos_dist*(20*np.pi/180)/(17-3.5),drone_pos_dist*(20*np.pi/180)/(17-3.5))
        drone_pitch = np.random.uniform(-drone_pos_dist*(10*np.pi/180)/(17-3.5),drone_pos_dist*(10*np.pi/180)/(17-3.5))

        x_offset = np.cos(drone_pos_ori)*drone_pos_dist + 3.5
        y_offset = np.sin(drone_pos_ori)*drone_pos_dist + np.random.uniform(1.0)
        z_offset = drone_height
        yaw_offset = drone_yaw+np.pi/2
        pitch_offset = drone_pitch

        gate_angle = gate_ori.as_euler('xyz')
        gate_angle[2] = gate_angle[2]+np.pi/2
        tmp = R.from_euler('xyz', gate_angle)

        offsets = tmp.apply(np.array([x_offset, y_offset, z_offset]))
        result_pos = -offsets + gate_pos
        if result_pos[2] > 15:
            result_pos[2] = 15
        gate_yaw = gate_ori.as_euler('xyz')[2]
        result_yaw = yaw_offset + gate_yaw
        result_pitch = pitch_offset

        return result_pos, np.array([0, result_pitch, result_yaw])

    def run(self, num_sample):
        data_fn = os.path.join(data_path, f"data.txt")
        with open(data_fn,'w+') as f:
            pass  
        kp_fn = os.path.join(data_path, 'kp.txt')
        with open(data_fn,'w+') as f:
            pass
        for i in range(num_sample):
            gate_idx = np.random.randint(len(self.gate_names_list))
            # gate_idx = 20
            gate_name = self.gate_names_list[gate_idx]
            
            gate_pose = self.gate_poses_ground_truth[gate_idx]

            gate_pos = np.array([
                gate_pose.position.x_val,
                gate_pose.position.y_val,
                gate_pose.position.z_val
            ])
            
            if gate_idx != 0:
                pre_gate_idx = gate_idx - 1
                pre_gate_pose = self.gate_poses_ground_truth[pre_gate_idx]
                pre_gate_pos = np.array([
                    pre_gate_pose.position.x_val,
                    pre_gate_pose.position.y_val,
                    pre_gate_pose.position.z_val
                ])
                gate_dist = np.linalg.norm(gate_pos - pre_gate_pos)
            else:
                gate_dist = 10

            gate_ori = [
                gate_pose.orientation.x_val,
                gate_pose.orientation.y_val,
                gate_pose.orientation.z_val,
                gate_pose.orientation.w_val,            
            ]

            inner_gd = self.airsim_client.simGetNominalGateInnerDimensions()
            outer_gd = self.airsim_client.simGetNominalGateOuterDimensions()

            gs = self.airsim_client.simGetObjectScale(gate_name)
            r:R = R.from_quat(gate_ori)
            yaw = r.as_euler('xyz')[2]
            gate_x_offset_inner = [np.cos(yaw)*inner_gd.x_val*gs.x_val, np.sin(yaw)*inner_gd.x_val*gs.x_val]
            gate_x_offset_outer = [np.cos(yaw)*outer_gd.x_val*gs.x_val, np.sin(yaw)*outer_gd.x_val*gs.x_val]
            gate_y_offset = [-np.sin(yaw)*inner_gd.y_val*gs.y_val/2, np.cos(yaw)*inner_gd.y_val*gs.y_val/2]
            gate_z_offset_inner = inner_gd.z_val*gs.z_val 
            gate_z_offset_outer = outer_gd.z_val*gs.z_val 
            gate_vert_inner = [
                [
                    gate_pos[0]+gate_x_offset_inner[0]-gate_y_offset[0], 
                    gate_pos[1]+gate_x_offset_inner[1]-gate_y_offset[1],
                    gate_pos[2]+gate_z_offset_inner
                ],
                [
                    gate_pos[0]-gate_x_offset_inner[0]-gate_y_offset[0], 
                    gate_pos[1]-gate_x_offset_inner[1]-gate_y_offset[1],
                    gate_pos[2]+gate_z_offset_inner
                ],
                [
                    gate_pos[0]-gate_x_offset_inner[0]-gate_y_offset[0], 
                    gate_pos[1]-gate_x_offset_inner[1]-gate_y_offset[1],
                    gate_pos[2]-gate_z_offset_inner
                ],
                [
                    gate_pos[0]+gate_x_offset_inner[0]-gate_y_offset[0], 
                    gate_pos[1]+gate_x_offset_inner[1]-gate_y_offset[1],
                    gate_pos[2]-gate_z_offset_inner
                ],
            ]    
            gate_vert_outer = [
                [
                    gate_pos[0]+gate_x_offset_outer[0]-gate_y_offset[0], 
                    gate_pos[1]+gate_x_offset_outer[1]-gate_y_offset[1],
                    gate_pos[2]+gate_z_offset_outer
                ],
                [
                    gate_pos[0]-gate_x_offset_outer[0]-gate_y_offset[0], 
                    gate_pos[1]-gate_x_offset_outer[1]-gate_y_offset[1],
                    gate_pos[2]+gate_z_offset_outer
                ],
                [
                    gate_pos[0]-gate_x_offset_outer[0]-gate_y_offset[0], 
                    gate_pos[1]-gate_x_offset_outer[1]-gate_y_offset[1],
                    gate_pos[2]-gate_z_offset_outer
                ],
                [
                    gate_pos[0]+gate_x_offset_outer[0]-gate_y_offset[0], 
                    gate_pos[1]+gate_x_offset_outer[1]-gate_y_offset[1],
                    gate_pos[2]-gate_z_offset_outer
                ],
            ]         

            pose = self.airsim_client.simGetVehiclePose(vehicle_name = self.drone_name)
            # pose.position.x_val = gate_pos[0]
            # pose.position.y_val = gate_pos[1]-17
            # pose.position.z_val = gate_pos[2]
            # pose.orientation.x_val = 0
            # pose.orientation.y_val = 0
            # pose.orientation.z_val = 0.7071068
            # pose.orientation.w_val = 0.7071068
            drone_pos, drone_ori = self.sample_drone_pose(gate_pos, r, gate_dist)
            pose.position.x_val = drone_pos[0]
            pose.position.y_val = drone_pos[1]
            pose.position.z_val = drone_pos[2]
            quat = R.from_euler('xyz', drone_ori).as_quat()
            pose.orientation.x_val = quat[0]
            pose.orientation.y_val = quat[1]
            pose.orientation.z_val = quat[2]
            pose.orientation.w_val = quat[3]
            self.airsim_client.simSetVehiclePose(pose = pose, vehicle_name = self.drone_name, ignore_collison=True)
            with open(data_fn, 'a') as f:
                f.write(f"{i},{pose.position.x_val},{pose.position.y_val},{pose.position.z_val},{pose.orientation.x_val},{pose.orientation.y_val},{pose.orientation.z_val},{pose.orientation.w_val}, {gate_idx}\n")
            with open(kp_fn, 'a') as f:
                kp_str = f'{i}, {gate_vert_inner[0][0]}, {gate_vert_inner[0][1]}, {gate_vert_inner[0][2]}'
                for j in range(1, len(gate_vert_inner)):
                    kp_str += f',{gate_vert_inner[j][0]}, {gate_vert_inner[j][1]}, {gate_vert_inner[j][2]}'
                for j in range(len(gate_vert_outer)):
                    kp_str += f',{gate_vert_outer[j][0]}, {gate_vert_outer[j][1]}, {gate_vert_outer[j][2]}'
                kp_str += f",{gate_idx}\n"
                f.write(kp_str)
            request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
            response = self.airsim_client_images.simGetImages(request)
            img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
            cv2.imshow('Object detector', img_rgb)
            cv2.waitKey(1)
            img_fn = os.path.join(img_path, f'./img_{i}.png')
            cv2.imwrite(img_fn, img_rgb)
            
        pass

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

def compute_accuracy(rm, dataset):
    num_accurate = 0
    total_points = 0
    max_rm = -float('inf')
    dist_list = []
    estimated_pos_list = []
    for data in dataset:
        detect_flag = data['detect_flag']
        if detect_flag:
            total_points += 1
            s = data['s']
            s_star = data['s_star']
            max_rm = max(max_rm, distance(s,s_star))
            dist = distance(s, s_star)
            dist_list.append(dist)

            a = data['a']
            a_star = data['a_star']
            xs = data['xs']
            drone_yaw = data['yaw']
            r = R.from_euler('xyz',[0,0,drone_yaw])
            estimated_pos_vec = np.linalg.inv(r.as_matrix())@(np.array(a)-np.array(xs))
            estimated_pos_list.append(estimated_pos_vec)

            if dist<rm:
                num_accurate += 1
    plt.hist(dist_list,bins=100)
    plt.savefig('tmp.png')    
    accuracy = num_accurate / total_points
    print(max_rm, total_points, num_accurate, accuracy)
    return np.array(estimated_pos_list)

def visualize(dataset, estimated_pos_list):
    plotter = pv.Plotter()
    estimated_pos = []
    for data in dataset:
        if data['detect_flag']:
            a = data['a']
            a_star = data['a_star']
            xs = data['xs']
            drone_yaw = data['yaw']
            r = R.from_euler('xyz',[0,0,drone_yaw])
            estimated_pos_vec = np.linalg.inv(r.as_matrix())@(np.array(a)-np.array(xs))
            estimated_pos.append(estimated_pos_vec.tolist())
    estimated_pos = np.array(estimated_pos)
    point_cloud = pv.PolyData(estimated_pos)
    plotter.add_mesh(point_cloud, color='b', point_size=10.,
                 render_points_as_spheres=True)

    real_pos_vec = np.linalg.inv(r.as_matrix())@(np.array(a_star)-np.array(xs))
    real_gate_pos = np.array([real_pos_vec.tolist()])
    point_cloud2 = pv.PolyData(real_gate_pos)
    plotter.add_mesh(point_cloud2, color='g', point_size=20.,
                 render_points_as_spheres=True)
    point_could3 = pv.PolyData(np.array([0,0,0]))
    plotter.add_mesh(point_could3, 'r', point_size = 20, render_points_as_spheres=True)

    # Compute center for the inferenced results
    center = np.mean(estimated_pos_list, axis = 0)
    diff = np.linalg.norm(center - real_pos_vec)
    radius_list = [] 
    for i in range(estimated_pos_list.shape[0]):
        radius_list.append(np.linalg.norm(estimated_pos_list[i,:]-center))
    radius_list = np.array(radius_list)
    radius_mean = np.mean(radius_list)
    print(">>> center", center)
    print(">>> diff", diff)
    print(">>> mean", radius_mean)
    radius_70percentile = np.percentile(radius_list, 70)
    print(">>> 70 percentile", radius_70percentile)
    radius_90percentile = np.percentile(radius_list, 90)
    print(">>> 90 percentile", radius_90percentile)
    sphere = pv.Sphere(center=center, radius=radius_70percentile)
    plotter.add_mesh(sphere, color='y', opacity=0.3)
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
    center = [-51.577632904052734, 112.41986083984375, 0.673719048500061]
    # center = [-51.330406188964844, 113.5843048095703, 0.2118362933397293]
    # center = [-51.577632904052734, 115.41986083984375, 0.673719048500061]
    # radius = np.array([1.0, 1.0, 1.0])
    # radius = np.array([3, 8.5, 2.5])
    # rect = [(np.array(center)-radius).tolist(), (np.array(center)+radius).tolist()]
    
    # # Specify the real position of the gate
    # gate_index = 6 
    # real_gate_pose:airsim.Pose = baseline_racer.gate_poses_ground_truth[gate_index]
    # real_gate_pos = [real_gate_pose.position.x_val, real_gate_pose.position.y_val, real_gate_pose.position.z_val]
    # real_gate_ori = [
    #     real_gate_pose.orientation.x_val, 
    #     real_gate_pose.orientation.y_val, 
    #     real_gate_pose.orientation.z_val, 
    #     real_gate_pose.orientation.w_val 
    # ]
    # r = R.from_quat(real_gate_ori)
    # gate_vector = np.array(center) - np.array(real_gate_pos)
    # gate_vector = np.linalg.inv(r.as_matrix())@gate_vector
    # gate_yaw = r.as_euler('xyz')[2]
    # yaw_vector = -1.371073 - gate_yaw
    # gate_vector = np.append(gate_vector, yaw_vector)
    # Specify the number of samples we want
    num_sample = 100

    # gate_idx_list = [0,1,2,3,4,5,6,7,8,9,11,12,15,16,17,18]
    # gate_idx_list = [6]

    # Randomly sample position and inference gate position
    res = baseline_racer.run(num_sample)
    # Compute the inferencing accuracy
    # estimated_pos_list = compute_accuracy(1, res)

    # visualize(res,estimated_pos_list)

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
        default="Qualifier_Tier_3",
    )
    args = parser.parse_args()
    main(args)
