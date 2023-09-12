import numpy as np 
import cv2
from scipy.spatial.transform import Rotation 
import os 
import matplotlib.pyplot as plt 
import json

script_dir = os.path.dirname(os.path.realpath(__file__))

def gaussian(xL, yL, sigma, H, W):

    grid = np.meshgrid(list(range(W)), list(range(H)))
    channel = np.exp(-((grid[0] - xL) ** 2 + (grid[1] - yL) ** 2) / (2 * sigma ** 2))

    return channel

def convertToHM(H, W, keypoints, sigma=5):
    nKeypoints = len(keypoints)

    img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

    for i in range(0, nKeypoints // 2):
        x = keypoints[i * 2]
        y = keypoints[1 + 2 * i]

        channel_hm = gaussian(x, y, sigma, H, W)

        img_hm[:, :, i] = channel_hm
    
    img_hm = img_hm.transpose((2,0,1))
    return img_hm

R_CV2NED = Rotation.from_matrix(np.asfarray([
    [ 0.,  1.,  0.],
    [-1.,  0.,  0.],
    [ 0.,  0.,  1.],
]))
rot_CV2CAM = Rotation.from_matrix([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
R_CAMCV = rot_CV2CAM.inv()

def project_3d_point_to_screen(subjectXYZ, camXYZ, camQuaternion, camProjMatrix4x4, imageWidthHeight):
    #Turn the camera position into a column vector.
    camPosition = np.transpose([camXYZ])

    #Convert the camera's quaternion rotation to yaw, pitch, roll angles.
    # pitchRollYaw = utils.to_eularian_angles(camQuaternion)
    r = Rotation.from_quat(camQuaternion)
    
    #Create a rotation matrix from camera pitch, roll, and yaw angles.
    camRotation = r.as_matrix() 
    
    #Change coordinates to get subjectXYZ in the camera's local coordinate system.
    XYZW = np.transpose([subjectXYZ])
    XYZW = np.add(XYZW, -camPosition)
    # print("XYZW: " + str(XYZW))
    XYZW = np.matmul(np.transpose(camRotation), XYZW)
    # print("XYZW derot: " + str(XYZW))
    
    #Recreate the perspective projection of the camera.
    XYZW = np.concatenate([XYZW, [[1]]])    
    XYZW = np.matmul(camProjMatrix4x4, XYZW)
    XYZW = XYZW / XYZW[3]
    
    #Move origin to the upper-left corner of the screen and multiply by size to get pixel values. Note that screen is in y,-z plane.
    normX = (1 - XYZW[0]) / 2
    normY = (1 + XYZW[1]) / 2
    
    return np.array([
        imageWidthHeight[0] * normX,
        imageWidthHeight[1] * normY
    ]).reshape(2,)

def convert_to_image(world_pos, ego_pos, ego_ori, img_dim, img_FOV):
    objectPoints = np.array(world_pos) 
    R = Rotation.from_quat(ego_ori)
    # # R_euler = R.as_euler('xyz')
    # # R_euler[1] = -R_euler[1]
    # # R = Rotation.from_euler('xyz', R_euler)
    # R_roted = R
    R2 = R_CAMCV
    R_roted = R2*R.inv()
    img_w, img_h = img_dim

    #TODO: The way of converting rvec is wrong
    rvec = R_roted.as_rotvec()
    tvec = -R_roted.apply(np.array(ego_pos))
    fx = fy = (img_w/2)/(np.tan(np.deg2rad(img_FOV)/2))
    cameraMatrix = np.array([
        [fx, 0.0, img_w/2], 
        [0.0, fy, img_h/2], 
        [0.0, 0.0, 1.0]
    ])
    distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    pnt,_ = cv2.projectPoints(objectPoints,rvec,tvec,cameraMatrix,distCoeffs)
    return pnt.reshape((2,))  

if __name__ == "__main__":
    with open(os.path.join(script_dir, './data/data.txt'),'r') as f:
        data = f.read()
        data = data.strip('\n').split('\n')
    with open(os.path.join(script_dir, './data/kp.txt'), 'r') as f:
        gate_data = f.read()
        gate_data = gate_data.strip('\n').split('\n')
    with open('/home/younger/Documents/AirSim/settings.json', 'r') as f:
        config = json.load(f)
    with open(os.path.join(script_dir, f"./label/img.txt"), "w+") as f:
        pass
    for idx in range(0, 100000):
        # idx = 1
        print(">>>", idx)
        # if idx == 6910:
        #     print("stop")

        pose = data[idx]
        pose = pose.split(',')
        pose = [float(elem) for elem in pose]

        gate_pos = gate_data[idx]
        gate_pos = gate_pos.split(',')
        gate_pos = [float(elem) for elem in gate_pos]

        # Generate intermediate keypoints
        gate_pos = np.array(gate_pos[1:-1])
        gate_pos = np.reshape(gate_pos, (-1,3))
        for j in range(4):
            tmp = (gate_pos[j,:] + gate_pos[(j+1)%4,:])/2
            gate_pos = np.vstack((gate_pos, tmp))
        for j in range(4):
            tmp = (gate_pos[4+j,:] + gate_pos[4+(j+1)%4,:])/2
            gate_pos = np.vstack((gate_pos, tmp))
        keypoint_list = gate_pos
        num_keypoint = keypoint_list.shape[0]
        
        drone_pos = np.array([pose[1], pose[2], pose[3]]) # x,y,z
        drone_ori = [pose[4], pose[5], pose[6], pose[7]] # x,y,z,w
        R = Rotation.from_quat(drone_ori)
        # drone_pos += offset_vec

        camera_projection_matrix = np.array([
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,10],
            [-1,0,0,0]
        ])

        cam_width = config['Vehicles']['drone_1']['Cameras']['fpv_cam']['CaptureSettings'][0]['Width']
        cam_height = config['Vehicles']['drone_1']['Cameras']['fpv_cam']['CaptureSettings'][0]['Height']
        cam_FOV = config['Vehicles']['drone_1']['Cameras']['fpv_cam']['CaptureSettings'][0]['FOV_Degrees']

        position_in_image = []
        for i in range(num_keypoint):
            # position_in_image.append(
            #     convert_to_image(keypoint_list[i], 
            #     drone_pos, 
            #     drone_ori,
            #     (cam_width, cam_height),
            #     cam_FOV
            # ))
            position_in_image.append(project_3d_point_to_screen(
                keypoint_list[i], 
                drone_pos,
                drone_ori,
                camera_projection_matrix,
                (cam_width,cam_height)
            ))

        u_vectors = []
        v_vectors = []
        kp_vectors = []
        for i in range(num_keypoint):
            u = int(position_in_image[i][0])
            v = int(position_in_image[i][1])

            u_vectors.append(u)
            v_vectors.append(v)

            kp_vectors.append(u)
            kp_vectors.append(v)

        img_fn = os.path.join(script_dir, f'./img/img_{idx}.png')
        img = cv2.imread(img_fn)

        cv2.circle(img, (u_vectors[0], v_vectors[0]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[1], v_vectors[1]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[2], v_vectors[2]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[3], v_vectors[3]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[4], v_vectors[4]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[5], v_vectors[5]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[6], v_vectors[6]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[7], v_vectors[7]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[8], v_vectors[8]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[9], v_vectors[9]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[10], v_vectors[10]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[11], v_vectors[11]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[12], v_vectors[12]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[13], v_vectors[13]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[14], v_vectors[14]), 2, (0,0,255))        
        cv2.circle(img, (u_vectors[15], v_vectors[15]), 2, (0,0,255))        
        # cv2.line(img, (u_vectors[0], v_vectors[0]), (u_vectors[1], v_vectors[1]),(0,0,255))
        # cv2.line(img, (u_vectors[1], v_vectors[1]), (u_vectors[2], v_vectors[2]),(0,0,255))
        # cv2.line(img, (u_vectors[2], v_vectors[2]), (u_vectors[3], v_vectors[3]),(0,0,255))
        # cv2.line(img, (u_vectors[3], v_vectors[3]), (u_vectors[0], v_vectors[0]),(0,0,255))

        # cv2.line(img, (u_vectors[4], v_vectors[4]), (u_vectors[5], v_vectors[5]),(0,0,255))
        # cv2.line(img, (u_vectors[5], v_vectors[5]), (u_vectors[6], v_vectors[6]),(0,0,255))
        # cv2.line(img, (u_vectors[6], v_vectors[6]), (u_vectors[7], v_vectors[7]),(0,0,255))
        # cv2.line(img, (u_vectors[7], v_vectors[7]), (u_vectors[4], v_vectors[4]),(0,0,255))

        # cv2.line(img, (u_vectors[8], v_vectors[8]), (u_vectors[9], v_vectors[9]),(0,0,255))
        # cv2.line(img, (u_vectors[9], v_vectors[9]), (u_vectors[10], v_vectors[10]),(0,0,255))
        # cv2.line(img, (u_vectors[10], v_vectors[10]), (u_vectors[11], v_vectors[11]),(0,0,255))
        # cv2.line(img, (u_vectors[11], v_vectors[11]), (u_vectors[8], v_vectors[8]),(0,0,255))

        # cv2.line(img, (u_vectors[12], v_vectors[12]), (u_vectors[13], v_vectors[13]),(0,0,255))
        # # cv2.line(img, (u_vectors[13], v_vectors[13]), (u_vectors[14], v_vectors[14]),(0,0,255))
        # # cv2.line(img, (u_vectors[14], v_vectors[14]), (u_vectors[15], v_vectors[15]),(0,0,255))
        # # cv2.line(img, (u_vectors[15], v_vectors[15]), (u_vectors[12], v_vectors[12]),(0,0,255))

        # plt.imshow(img)
        # plt.show()
        # cv2.imshow('keypoints', img)
        # cv2.imshow('keypoints', img)
        # cv2.waitKey(3)
        hms = convertToHM(cam_width, cam_height, kp_vectors, sigma=2)
        # hm_img = np.vstack((np.hstack(hms[0:4,:,:]), np.hstack(hms[4:,:,:])))
        hm_img = np.sum(hms, axis=0)
        # cv2.imshow('heat map', hm_img)
        # cv2.waitKey(1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(script_dir, f"./label/img_marker_{idx}.png"), img)
        print("HM shape: ", hms.shape[0])
        for j in range(hms.shape[0]):
            cv2.imwrite(os.path.join(script_dir, f"./label/img/img_{idx}_{j}.png"), hms[j,:,:]*255)

        cv2.imwrite(os.path.join(script_dir, f"./label/img/img_{idx}_hm.png"), hm_img*255)

        with open(os.path.join(script_dir, f"./label/img.txt"), "a") as f:
            f.write(f'{idx},')
            for i in range(num_keypoint):
                f.write(f"{u_vectors[i]},{v_vectors[i]},")
            f.write('\n')
            # f.write(f"{u_vectors[0]}, {v_vectors[0]}, {u_vectors[1]}, {v_vectors[1]}, {u_vectors[2]}, {v_vectors[2]}, \
            #         {u_vectors[3]}, {v_vectors[3]}, {u_veimg_rgbctors[4]}, {v_vectors[4]}, {u_vectors[5]}, {v_vectors[5]}, \
            #         {u_vectors[6]}, {v_vectors[6]}, {u_vectors[7]}, {v_vectors[7]}, {u_vectors[8]}, {v_vectors[8]}, \
            #         {u_vectors[9]}, {v_vectors[9]}, {u_vectors[10]}, {v_vectors[10]}, {u_vectors[11]}, {v_vectors[11]},\
            #         {u_vectors[12]}, {v_vectors[12]}, {u_vectors[13]}, {v_vectors[7]}, {u_vectors[8]}, {v_vectors[8]},\
            #         ")
        # import matplotlib.pyplot as plt 
        # plt.imshow(img_rgb)
        # plt.show()
