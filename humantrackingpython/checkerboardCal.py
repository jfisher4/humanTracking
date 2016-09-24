import numpy as np
import cv2
import glob
import pickle

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)#32
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#worldPoints=pickle.load( open( "worldPoints.p", "rb" ) ) #code for loading world points from file. 
#imagePoints=pickle.load( open( "imagePoints.p", "rb" ) ) #code for loading world points from file. 
obj_points = np.array(objpoints, 'float64')
img_points = np.array(imgpoints, 'float64')
images = glob.glob('*.jpeg')
print(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #height, width, depth = img.shape

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname)
        objpoints.append(objp)
        
        cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners,ret)
        
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
print(gray.shape[::-1])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(len(objpoints))
print(len(imgpoints))

#code for object height and stuff
#retval, rvec, tvec = cv2.solvePnP(obj_points,img_points,mtx,dist)
#rotM, other = cv2.Rodrigues(rvec)
#pickle.dump(rotM, open( "rotM.p", "wb" ) )
#save the camera matrix
pickle.dump(mtx, open( "camera_matrix.p", "wb" ) )
pickle.dump(dist, open( "dist.p", "wb" ) )
pickle.dump(tvecs, open( "tvecs.p", "wb" ) )
pickle.dump(rvecs, open( "rvecs.p", "wb" ) )
#pickle.dump(tvec, open( "tvec.p", "wb" ))

#optimize camera matrix
#img = cv2.imread('20150510_163602.jpg')
#h,  w = img.shape[:2]
#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#pickle.dump(newcameramtx, open( "newcamera_matrix.p", "wb" ) )

# undistort
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#print(dst)
# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)


# calculate the reprojection error, should be close to zero
mean_error = 0
tot_error = 0
print(len(objpoints))
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error #tot_error
    print(error)

print "total error: ", mean_error/len(objpoints)

