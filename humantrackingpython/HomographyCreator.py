import pickle
import cv2
import numpy as np
import os
from math import sqrt


# Global variables.
gx=0
gy=0
visible=False
init=False



# Class that handles all aspects of training an HOG descriptor, from creating sample images to actual SVM training.
class CreateHomography():

# Initializes the class.    
    def __init__(self, directory):
        self.currentDirectory=directory
        self.positiveSampleImages=0
        self.negativeSampleImages=0
        os.chdir(directory)
        self.homography = []
        print(os.getcwd())
       
        

# Helper function that listens for mouse actions for the VideoCropperHelper method. also for manually dragging box over a person in a video or image.   
    def draw_roi(self,event,x,y,flags,param):
        global gx,gy,gx1,gy1,visible,init, clicked, left, right
        if event == cv2.EVENT_RBUTTONDOWN:
            #global gx,gy,visible,init
            gx=x
            gy=y
            gx1 = x
            gy1 = y
            visible=True
            init=True
            left = False
            clicked = True
            right = True
            
        if clicked is True and right is True: #Not working
            if cv2.EVENT_MOUSEMOVE: 
                print('test')
                gx1=x
                gy1=y
                visible=True
                init=True
            
        if event == cv2.EVENT_LBUTTONDOWN:
            gx=x
            gy=y
            gx1 = -1
            gy1 = -1
            visible=True
            init=True
            left = True
            right= False
         
        if event == cv2.EVENT_RBUTTONUP:
            #global gx1,gy1,visible,init
                       
            gx1=x 
            distX = int(round(sqrt((gx1 - gx)**2))) # use numpy functions instead
            gy1 = gy+ (2*distX)
            visible=True
            init=True
            clicked = False
            right = False
            
       

# Cycles through video files in working directory to extract positive and negative sample images from.        
    #def VideoCropper(self):
        #files = raw_input("Enter name of file: ") #raw_input("Enter name of image: ") #VIRAT_S_000005_4758.jpeg
        #self.VideoCropperHelper(os.getcwd()+"\\"+files)
        #for files in os.listdir(os.getcwd()):
            #print files
            #if files.endswith(".mp4") or files.endswith(".wmv") or files.endswith(".avi") or files.endswith(".MOV") or files.endswith(".MTS"): 
                #print(os.getcwd()+"\\"+files)
                #self.VideoCropperHelper(os.getcwd()+"\\"+files)
            #else:
                #continue
            
# Helper function that processes video with openCV in order to extract positive and negative sample images.
# Keyboard Commands:
# =========================================================================================================
# V = Next frame
# N = Save as negative image
# SpaceBar = Save as positive image
# S = Save video image for Homography
# < = Decrease roi
# > = Increase roi
# Q = Quit
    def VideoCropper(self): #Helper
        #filename = raw_input("Enter name of file: ") #raw_input("Enter name of image: ") #VIRAT_S_000005_4758.jpeg
        filename = '01072016A5_F1.mp4'
        #filename = '00018.MTS'
        filename = os.getcwd()+"\\"+filename
        print("Select points in same order as they appear in corresponding world coordinate array.")
        print("Click left mouse to select point,[ h = add point to list," )
        print(" d = remove a point from the list, w = select first point")
        print("for height measurement, e = select second point for height")
        print("measurement, j = compute homography, p = find world coord.,")
        print(" l = load from previous saved homography, c = Next frame,")
        print(" n = save neg image, ' ' = save pos image, s = save frame,")
        print(" < = decrease roi, > = increase,v = save vectors,z = calc heading")
        print(" q = quit]")
        try:
            self.homography = pickle.load( open( "01072016A5_H.p", "rb" ) ) #code for loading homography from file.
            print("homograpy successfully loaded")
        except:
            print("no previously saved homography exists, you must create the homography")
        cv2.namedWindow('frame')#, flags = cv2.WINDOW_NORMAL)# toggle flag when using on a limited resolution display. not for exact measurements though. 
        
        if filename.endswith(".mp4") or filename.endswith(".wmv") or filename.endswith(".avi") or filename.endswith(".MOV") or filename.endswith(".MTS"): 
            cap = cv2.VideoCapture(filename)
            #print(cap)
            jmpframe=int(raw_input("Enter frame number to start(0 to start at beginning):"))    
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            cap = cv2.imread(filename,1)
            jmpframe = 0

        global xA,yA,x,y,gx,gy,gx1,gy1,visible,init, clicked, left, vectorList, imagePoints
        gx=-1
        gy=-1
        gx1=-1
        gy1=-1
        x = "None"
        y = "None"
        xA = "None"
        yA = "None"
        visible=False
        init=False
        clicked = False
        left = False
        imagePoints = []
        vectorList = []
        ret,frameClean=cap.read() #save first frame for background subtraction
        frameno=0
        while (frameno!=jmpframe): # skips to chosen frame
            ret,frame=cap.read()
            frameno=frameno+1
        try:
            status = cap.isOpened()
        except:
            status = cap is not None
        frameno=frameno+1
        while(status):          #cap.isOpened()): or cap is not None
            width=64
            height=128
            try:
                ret, frame = cap.read() # for video files
            except:
                frame = cap  # for image files
            #print(frame)
            tmpframe=np.array(frame)
            #print(tmpframe)
            cv2.setMouseCallback('frame',self.draw_roi) # listens for mouse events
            print("Frame number: "+ str(frameno))
            
            while(True):
                
                cv2.imshow('frame',tmpframe)

                if (visible):
                    if(init):
                        tmpframe=np.array(frame)
                        
                        #cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                        init=False  
                if gx != -1 and gx1 == -1:# and left is True: 
                    cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                    #gx1 = -1
                    #left = False
                if gx != -1 and gx1 != -1:
                    cv2.rectangle(tmpframe,(gx,gy),(gx1,gy1),(0,255,0),1)
                    #gx1 = -1
                    #if clicked:
                        #break
                k = cv2.waitKey(10) & 0xFF
                
                    
                if k== ord('h') and visible and gx != "None":
                        cv2.circle(tmpframe,(gx,gy),2,(255,0,0), 2) 
                        cv2.putText(tmpframe, str(gx)+','+str(gy),(gx+10,gy), 0, 1, (255,0,0), 1,8, False)
                        imagePoints.append([gx,gy])
                        init=False
                        print(imagePoints)

                if k== ord('p') and visible and gx!= "None":
                    cv2.circle(tmpframe,(gx,gy),2,(255,0,0), 2) 
                    coordText = HomographyCreator.PointToWorld((gx,gy))
                    cv2.putText(tmpframe, str(coordText),(gx+10,gy), 0, 1, (255,0,0), 1,8, False)
                    init=False
                    print(coordText)

                if k== ord('l') and visible: #and init and x != "None":  attempt to load prior homography
                    try:
                        self.homography = pickle.load( open( "01072016A5_H.p", "rb" ) ) #code for loading homography from file.
                        print('homography.p successfully loaded')
                    except:
                        print('homography not available')
                    init=False
                    
                if k== ord('w') and visible and gx!= "None":
                    print("First coordinate selected.")
                    x,y = HomographyCreator.PointToWorld((gx,gy))
                    xA,yA = gx,gy
                    init=False
                    
                if k== ord('e') and visible and x != "None": # calculate object height
                    print('Second coordinate selected')
                    x2,y2 = HomographyCreator.PointToWorld((gx,gy))
                    bottomPoint = x,y
                    topPoint = x2,y2
                    height1 = HomographyCreator.ObjectHeight(bottomPoint,topPoint)
                    print(height1)
                    cv2.line(tmpframe,(xA,yA),(gx,gy),(255,0,0), 2) 
                    cv2.putText(tmpframe, str(height1),(gx+10,gy), 0, 1, (255,0,0), 1,8, False)
                    x = "None"
                    y = "None"
                    gx = -1
                    gy = -1
                    xA = "None"
                    yA = "None"
                    init=False
                    
                if k== ord('j') and visible:
                    #imagePoints = [[945,990], [49,608], [540,699], [342,643], [1459,829], [304,592], [892,659], [638,620], [1725,738], [525,580]]#football field SN186
                    #imagePoints = np.array(imagePoints, 'float32') #float64
                    pickle.dump(imagePoints, open( "01072016A5_ImgPoints.p", "wb" ) )
                    #imagePoints=pickle.load( open( "imagePoints.p", "rb" ) ) #code for loading world points from file.
                    #worldPoints = [(691812.05,3883392.82,0),(691801.44,3883386.12,0),(691822.43,3883410.08,0),(691807.13,3883400.56,0),(691796.62,3883393.87,0),(691781.15,3883384.21,0),(691817.60,3883417.84,0),(691776.31,3883391.94,0)]#first filming camera A 
                    #worldPoints = [(691757.48,3883455.96,0),(691768.00,3883462.41,0),(691747.05,3883438.48,0),(691762.40,3883448.23,0),(691772.88,3883454.93,0),(691788.20,3883464.28,0),(691751.99,3883430.72,0),(691793.07,3883456.56,0)]#first filming camera B
                    #worldPoints = [(692110.58,3883007.73,0),(692108.86,3883014.08,0),(692103.33,3883024.27,0),(692114.68,3883067.53,0),(692144.08,3883026.75,0)] #quad area 102915 cam a
                    #worldPoints = [(691794.84,3883419.91,0),(691812.62,3883425.63,0),(691817.54,3883417.90,0),(691804.81,3883404.44,0),(691799.80,3883412.13 ,0)] #11032015CamA
                    #worldPoints = [(691807.80,3883433.26,0),(691805.27,3883437.29,0),(691787.49,3883431.53,0),(691790.00,3883427.65,0),(691794.92,3883419.89 ,0)] #12102015CamA 1.6 zoom
                    worldPoints = [(691816.86,3883419.30,0),(691802.73,3883408.18,0),(691800.39 ,3883412.01 ,0),(691797.95,3883415.86,0),(691813.98,3883423.90,0)] # 01072016CamA 1.6 zoom
                    #worldPoints = np.array(worldPoints, 'float32') #need extra precision to keep integrity of floats
                    pickle.dump(worldPoints, open( "01072016A5_WrldPoints.p", "wb" ) )
                    if len(imagePoints) >= 4 and len(imagePoints) == len(worldPoints):                    
                        print("Computing homography matrix of ground plane.")
                        #global worldPoints
                        #worldPoints=pickle.load( open( "worldPoints.p", "rb" ) ) #code for loading world points from file.
                        worldPoints1 = []
                        for i in worldPoints:
                            worldPoints1.append(i[:2])
                        HomographyCreator.FindHom(imagePoints, worldPoints1)
                        #pickle.dump(worldPoints, open( "worldPoints_A.p", "wb" ) )
                    else:
                        print("not enough image points to calculate homography")
                
                if k== ord('d') and visible and len(imagePoints) > 0:
                    print(imagePoints.pop(-1))
                    print(imagePoints)
                    print("Removed last chosen pixel homography point.")
                    
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a positive image.
                if k== ord(' ') and visible:
                    print("Positive")
                    roi=frame[gy:gy+height,gx:gx+width]
                    resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+ "_" + str(gx)+ "_" + str(gy) + "_P.jpeg"  
                    cv2.imwrite(saveName,resized_roi)
                    break
                
# Save Image for Homography 
                if k== ord('s') and visible:
                    print("Homography Image")
                    #print(frame)
                    #roi=frame[gy:gy+height,gx:gx+width]
                    #resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+".jpeg"  
                    cv2.imwrite(saveName, frame)                    
                    break
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a negative image.
                if k== ord('n') and visible:
                    print("Negative")
                    roi=frame[gy:gy+height,gx:gx+width]
                    resized_roi = cv2.resize(roi, (64,128))
                    saveName=filename[0:-4]+"_"+str(frameno)+ "_" + str(gx)+ "_" + str(gy) + "_N.jpeg"  
                    cv2.imwrite(saveName,resized_roi)                    
                    break
# Decreases size of roi.
                if k== ord(','):
                    if width>=16:
                        width=width-1
                        height=height-2
                        tmpframe=np.array(frame)
                        cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                        print("Width:"+str(width)+ " Height:"+str(height))
# Increases size of roi.
                if k== ord('.'):
                    if width<=128:
                        width=width+1
                        height=height+2
                        tmpframe=np.array(frame)
                        cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                        print("Width:"+str(width)+ " Height:"+str(height))
# Skips to next frame. and/or clears screen
                if k== ord('c'):
                    break
 # saves vectors to list in world coordinates.                
                if k== ord('v')and gx1 != -1: #use right click and drag box over person.
                    #calculate the geometric center
                    #gxCenter = (gx + gx1)/2
                    #gyCenter = (gy +gy1)/2
                    dx,dy = HomographyCreator.PointToWorld((gx,gy1))#initial values before transformation to world 
                    ax,ay = HomographyCreator.PointToWorld((gx,gy))
                    cx,cy = HomographyCreator.PointToWorld((gx1,gy1))
                    ex = (dx+cx)/2
                    ey = (dy+cy)/2
                    #ex,ey =  HomographyCreator.PointToWorld((gxCenter,gyCenter))
                    height1 = HomographyCreator.ObjectHeight((dx,dy),(ax,ay))
                    a = [dx,dy,height1] #points in 3D world
                    b = [cx,cy,height1]
                    c = [cx,cy,0]
                    d = [dx,dy,0]
                    e = [ex,ey,height1/2] #center of person
                    vectorList.append([frameno,a,b,c,d,e])
                    pickle.dump(vectorList, open( "vectorList.p", "wb" ) )
                    gx1 = -1
                    gy1 = -1
                    gx = -1
                    gy = -1
                    break
                
#z for direction
                if k== ord('z'): 
                    vectorList = pickle.load( open( "vectorList.p", "rb" ) )
                    heading = self.findHeading(vectorList)
                    print(heading)
                
# q for quit            
                if k== ord('q'):
                    try: 
                        cap.release()
                    except:
                        cap = None
                    try:
                        status = cap.isOpened()
                    except:
                        status = cap is not None
                    break

            visible=False
            frameno=frameno+1
        try: 
            cap.release()
        except:
            cap = None
        cv2.destroyAllWindows()

    def DisplayOptions(self):
        print("=============================================================================================")
        print("HOGModule for use in ML training, creating samples, and extracting feature vectors.")
        print("----Options----")
        print("1. Display amount of samples in directory")
        print("2. Create feature vectors from samples in directory")
        print("3. Test and train the SVM")
        print("4. Interact with a video or picture")
        print("5. View the svm model on a video.")
        print("6. Exit")
      
#code for findind the direction of movement
    def findHeading(self, vectorList):
        headingList = []
        if len(vectorList) > 1:
            #d0
            #worldPoints=pickle.load( open( "worldPoints.p", "rb" ) ) #code for loading world points from file.
            #origin = worldPoints[0]
            for i in range(len(vectorList)-1):
                frame1 = vectorList[i+1][0]
                vertex0 = vectorList[i][5] #fix math for 2D vector
                vertex1 = vectorList[i+1][5]
                #legA = abs(vertex1[0] - vertex0[0])
                #legB = abs(vertex1[1] - vertex0[1])
                #angleBeta = np.arctan((np.divide(legB,legA)))
                #angleBetaDeg = np.rad2deg(angleBeta)
                dx = vertex1[0] - vertex0[0]
                dy = vertex1[1] - vertex0[1]
                if dx > 0:
                    bearing = np.subtract(90,np.rad2deg(np.arctan((np.divide(dy,dx)))))#90
                elif dx < 0:
                    bearing = np.subtract(270,np.rad2deg(np.arctan((np.divide(dy,dx)))))#270
                else: #dx == 0
                    if dy > 0:
                        bearing = 0
                    elif dy < 0:
                        bearing = 180
                    else: # dy == 0
                        continue #points are the same no change in bearing
                
                print('The heading for frame No. %d is %0.2f degrees.' % (frame1, bearing))#angleBetaDeg))
                headingList.append(bearing)#angleBetaDeg
                headingAvg = 0
                for i in range(len(headingList)):
                    headingAvg += headingList[i]
                headingAvg = headingAvg / len(headingList)
                print('The average heading is %0.2f degrees.' %headingAvg)
        else:
            print('Not enough entries present to calculate a vertex')
      
# code for finding an objects height in world coordinates
    def ObjectHeight(self,bottomPoint,topPoint):
        try:
            worldPoints=pickle.load( open( "01072016A5_WrldPoints.p", "rb" ) ) #code for loading world points from file. 
            imagePoints=pickle.load( open( "01072016A5_ImgPoints.p", "rb" ) ) #code for loading world points from file. 
            obj_points = np.array(worldPoints, 'float64')
            img_points = np.array(imagePoints, 'float64')
            camera_matrix = pickle.load( open( "camera_matrix_A1_6.p", "rb" ) )
            dist_coefs = pickle.load( open( "dist_A1_6.p", "rb" ) )
            retval, rvec, tvec = cv2.solvePnP(obj_points,img_points,camera_matrix,dist_coefs)
            rotM, other = cv2.Rodrigues(rvec)
            pickle.dump(rotM, open( "01072016A5_rotM.p", "wb" ) )
            pickle.dump(tvec, open( "01072016A5_tvec.p", "wb" ) )
            cameraPosition = -np.matrix(rotM).T * np.matrix(tvec) #perhaps pickle.dump this for future use
            cameraHeight = cameraPosition[2]
            bottom = np.array((bottomPoint[0] ,bottomPoint[1], 0))#or use z = 1 if trouble add one to camera height 
            top = np.array((topPoint[0], topPoint[1], 0))#or use z = 1
            cameraBase = np.array((cameraPosition[0],cameraPosition[1], 0))#or use z = 1
    #1 find distance from base of camera to top point
            cameraBaseToTop = np.linalg.norm(cameraBase-top)
    #2 find angle between line from base of camera to top toint, and line from top point to camera, assume camera is at 90deg angle from ground creating a right triangle.
            radAlpha = np.tan(np.divide(cameraHeight,cameraBaseToTop))
    #3 find distance from bottom point to top point,
            bottomToTop = np.linalg.norm(bottom-top)
    #4 use the angle and the distance in step 3 to calculate height of object.
            tanAlpha = np.tan(radAlpha) #figure out why tan doesnt return expected valuse, has to do with degrees vs rad.
            objectHeight  =  np.multiply(bottomToTop, tanAlpha) #tanTheta)
            return objectHeight
        except:
            print("required files are not present worldPoints.p,imagePoints.p,camera_matrix.p,dist.p")
            #old code for estimating camera intrinsics/extrinsics
            #w = 1920#1920#1280#
            #h = 1440#1080#720#
            #size = (w,h)
            #retval,camera_matrix,dist_coefs,rvecs,tvecs = cv2.calibrateCamera([obj_points],[img_points],size,flags= cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO ) #+ cv2.CALIB_FIX_K3)#cv2.CALIB_USE_INTRINSIC_GUESS)#cv2.CALIB_FIX_K3, cv2.CALIB_FIX_ASPECT_RATIO 

# code for calculating world coord from pixel coord
    def PointToWorld(self,imagePoint):
        if not self.homography == []:
            point = ([[imagePoint[0]],[imagePoint[1]],[1]])
            pointMatrix = np.array(point,dtype=np.float32)
            worldXyz = np.dot(self.homography, pointMatrix) #dot or multiply?
            world = np.divide(worldXyz,worldXyz[2])
            worldX, worldY = world[0],world[1]
            #worldVal = str(world[0])+","+str(world[1])
            #print(worldX,worldY)
            return (worldX, worldY)
        else:
            print("You must first create homography matrix.")
        
#code for calculating homography matrix
    def FindHom(self,imagePoints,worldPoints): # need to verify that the matrix is correct
        """Calculates the homography if there are 4+ point pairs"""
        src = np.array(imagePoints, dtype=np.float64)      # using cv2.getPerspectiveTransform returns the same matrix as the code above but only allows for 4 points.
        dest = np.array(worldPoints, dtype=np.float64)
        H3, mask = cv2.findHomography(src, dest, cv2.RANSAC,5.0)
        self.homography = H3
        origin = ([[imagePoints[0][0]],[imagePoints[0][1]],[1]])
        originMatrix = np.array(origin,dtype=np.float32)
        originTest = np.dot(H3, originMatrix)
        originTestXY = np.divide(originTest,originTest[2])
        pickle.dump(H3, open( "01072016A5_H.p", "wb" ) )
        print("Matrix H3 derived from cv2.findHomography(src, dest, cv2.RANSAC,5.0).")
        print(H3)
        print("Image coordinates of world origin.")
        print(originMatrix)
        print("Origin test result world coordinates.")
        print(originTestXY)
          
# Start of script.    
if __name__ == '__main__':
    #inputDir=raw_input("Enter the name of the directory where the files are located for HOG Training:\n")
    inputDir = 'C:\\Users\\James\\Documents\\Millennium\\Pinos\\multiview'
    #inputDir = 'C:\\Users\\James\\Documents\\Millennium\\Pinos\\CrowdDynamics\\CameraA'
    HomographyCreator=CreateHomography(inputDir)
    end=False
    HomographyCreator.DisplayOptions()
    while(not end):
        inputCommand=raw_input("Ready for command: ")
        if (inputCommand=="1"):
            HomographyCreator.HOGSampleCount()
            HomographyCreator.DisplayOptions()            
        elif(inputCommand=="2"):
            HomographyCreator.FeatureVectorCreator()
            HomographyCreator.DisplayOptions()
        elif(inputCommand=="3"):
            HomographyCreator.Train_And_Test()
            HomographyCreator.DisplayOptions()
        elif(inputCommand=="4"):
            HomographyCreator.VideoCropper()
            HomographyCreator.DisplayOptions()
        elif(inputCommand=="5"):
            HomographyCreator.ViewSVMVideo()
            HomographyCreator.DisplayOptions()
        elif(inputCommand=="6"):
            end=True
        else:
            print("Not a valid command. Try Again...")
            HomographyCreator.DisplayOptions()
    print("Ending program...")
else:
    print 'Homography Creator is being imported from another module.'

