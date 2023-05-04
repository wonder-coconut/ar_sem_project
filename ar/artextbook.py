import cv2
import numpy as np
import math
import os
from objloader import *

camera_parameters_webcam = np.array([[1.41405685e+03, 0.00000000e+00, 6.59924138e+02],[0.00000000e+00, 1.42191604e+03, 4.72514351e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#phones are shot at 30 fps
camera_parameters_phone = np.array([[3.31609622e+03, 0.00000000e+00, 2.00609075e+03],[0.00000000e+00, 3.09535346e+03, 1.48778782e+03],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

camera_parameters = camera_parameters_phone
DEFAULT_COLOR = (0, 0, 0)

def main():
    #initialize homography
    homography_matrix = None
    #webcam stream capture
    videocap = cv2.VideoCapture("../assets/videos/4phone.mp4")
    #videocap = cv2.VideoCapture(0)
    videocap.set(3,1280)
    videocap.set(4,720)
    #image target

    scale = 50

    dir_name = os.getcwd()

    imgTargets = []
    hT = []
    wT = []
    kp1 = []
    des1 = []
    orb = cv2.ORB_create(nfeatures=2000)
    image_dir = os.path.join(dir_name, '../assets/images/textbook_test')
    
    targets = 0
    for image in os.listdir(image_dir):
        targets += 1
        imgTargets.append(cv2.imread(f'{image_dir}/{image}',0))
        width = int(imgTargets[-1].shape[1] * scale / 100)
        height = int(imgTargets[-1].shape[0] * scale / 100)
        dsize = (width,height)
        imgTargets[-1] = cv2.resize(imgTargets[-1], dsize)
        ht, wt = imgTargets[-1].shape
        kp, des = orb.detectAndCompute(imgTargets[-1],None)
        hT.append(ht)
        wT.append(wt)
        kp1.append(kp)
        des1.append(des)

    #3d model
    obja = OBJ(os.path.join(dir_name, '../assets/models/rat.obj'), swapyz=True)
    objb = OBJ(os.path.join(dir_name, '../assets/models/fox.obj'), swapyz=True)
    
    bf = cv2.BFMatcher()   
    framecount = 0 

    while True: #framewise detection
        framecount += 1
        success, imgWebCam = videocap.read()
        kp2, des2 = orb.detectAndCompute(imgWebCam,None)
        matches_targets = []
        
        i = 0
        while(i < targets):
            matches_targets.append(bf.knnMatch(des1[i],des2,k=2))
            i += 1
        
        goodMatches_targets = []
        for matches in matches_targets:
            goodMatches = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    goodMatches.append(m)
            goodMatches_targets.append(goodMatches)

        i = 0
        for goodMatches in goodMatches_targets:
            if(len(goodMatches) > 20):
                srcPts = np.float32([kp1[i][m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
                dstPts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)
                homography_matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
                pts = np.float32([[0,0],[0,hT[i]],[wT[i],hT[i]],[wT[i],0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, homography_matrix)
                frame = cv2.polylines(imgWebCam,[np.int32(dst)],True,(255,0,0),3)
                print(f"{framecount} image {i} detected")
            i += 1
        try:
            cv2.imshow('frame',frame)
        except:
            pass
        

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    videocap.release()
    cv2.destroyAllWindows()
    return 0


def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            #print(f"rendering face : {face}")
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            #print(f"rendering face : {face}")
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
        
    return img #ONE SINGLE COPY PASTE ERROR SET ME BACK TWO WEEKS IN MY GODDAMN PROJECT I WILL SHOOT SOMEONE I CANT ANYMORE

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

if __name__ == '__main__':
    main()