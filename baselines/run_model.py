import tensorflow.compat.v1 as tf
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
import numpy as np 
import json 
from scipy.spatial.transform import Rotation as R

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
        #     print("=============== NOT DETECT ===============")
    else:
        # print('==================== set detect_flag to FALSE ====================')
        estimate_depth = 8
        detect_flag = False
    return H, W, My, Mx

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [roll, pitch, yaw]

def main(dir):
    max_e_yaw = -float('inf')
    for idx in range(270, 280, 3):
        print(f"### {idx}")
        fn_img = os.path.join(dir, f'{idx}.jpg')
        image_rgb = cv2.imread(fn_img)
        print(type(image_rgb))
        H, W, My, Mx = inference(image_rgb)
        print(f"Inferred - H: {H}, W: {W}, My: {My}, Mx: {Mx}")

        d_hat = None
        e_yaw = None 
        e_pitch = None
        if H is not None:
            # Estimated distance between drone and gate 
            K1 = 13.401
            K2 = -1.976
            d_hat = K1*np.exp(K2*W)

            ey = (20 * (Mx - 0.5))
            e_yaw = np.arctan2(ey, d_hat)
            # max_e_yaw = max(max_e_yaw, abs(e_yaw))

            # Estimated pitch error between drone and gate
            ez = (20 * (My - 0.5))
            e_pitch = np.arctan2(ez, d_hat)

        fn_json = os.path.join(dir, f'{idx}.json')
        with open(fn_json,'r') as f:
            res = json.load(f)
        H, W, My, Mx = res["H"], res["W"], res['My'], res['Mx']
        print(f"Json     - H: {H}, W: {W}, My: {My}, Mx: {Mx}")
        print(f"distance: {d_hat}, pitch error: {e_pitch}, yaw error: {e_yaw}")

        w,x,y,z = res['drone_ori'][0], res['drone_ori'][1], res['drone_ori'][2], res['drone_ori'][3]
        r = R.from_quat([x,y,z,w])

        drone_x, drone_y, drone_z = res['drone_pos'][0], res['drone_pos'][1], res['drone_pos'][2]
        if d_hat is not None:
            # roll_gate = r.as_euler('xyz')[0]
            # pitch_gate = r.as_euler('xyz')[1]+e_pitch
            # yaw_gate = r.as_euler('xyz')[2]+e_yaw
            # r_gate = R.from_euler('xyz',[roll_gate, pitch_gate, yaw_gate])
            # estimated_gate_pose = r_gate.apply([d_hat,0,0]) + np.array([drone_x, drone_y, drone_z])
            gate_pos_drone = [d_hat, ey, ez]
            estimated_gate_pose = r.apply(gate_pos_drone) + np.array([drone_x, drone_y, drone_z])
            print(drone_x, drone_y, drone_z)
            print(estimated_gate_pose)
    print(max_e_yaw)

if __name__ == "__main__":
    fn = './picture_sc/picture_sc_10_11_3/'
    main(fn)