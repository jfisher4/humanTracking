import cv2
import numpy as np
#from matplotlib import path, transforms
from imutils.object_detection import non_max_suppression
from sklearn.cluster import DBSCAN


def bsRoiFinderPerson(person,ROIs,fgmask,img,ROI_RESIZE_DIM): #returns only one BS roi for a person
    matches = []
    box1 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
    for box in ROIs:
        box2 = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        lapping = overLap(box1,box2) 
        #print(box)
        #print(lapping,' person OverLap') 
        if lapping > 0:
            roiHist = roiFGBGHist(img,fgmask,box[0],box[1],box[2],box[3],ROI_RESIZE_DIM)
            histDist = histogramComparison(person.roiHist,roiHist)
            #print(histDist,'histdist')
            if len(matches)>0:
                if histDist < matches[0][2]:
                    matches = [(lapping, box, histDist)]
                elif lapping >= matches[0][0]:
                    matches = [(lapping, box, histDist)]
            else: #used in first iteration to set up overlap and histogram comparison
                matches = [(lapping, box, histDist)]
        
    if len(matches) > 0:
        #print(list(matches[0][1]),'selected boxes in bsroifinder')
        return list(matches[0][1])
    else:
        #print([],'selected boxes in bsroifinder')
        return []


def bsRoiFinderBox(box0,roiHist,ROIs,fgmask,img,ROI_RESIZE_DIM): #returns only one BS roi for a person
    matches = []
    box1 = [box0[0], box0[1], box0[0]+box0[2], box0[1]+box0[3]]
    for box in ROIs:
        box2 = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        lapping = overLap(box1,box2) #largest overlap
        #print(box1,box2)
        #print(lapping,' person OverLap') 
        if lapping > 0:
            roiHist2 = roiFGBGHist(img,fgmask,box[0],box[1],box[2],box[3],ROI_RESIZE_DIM)
            histDist = histogramComparison(roiHist,roiHist2)
            if len(matches)>0:
                if lapping >= matches[0][0]:
                    if histDist < matches[0][2]:
                        matches = [(lapping, box, histDist)]  #flag of one means it was found in  lost people
                    else:
                        matches = [(lapping, box, histDist)]
            else: #used in first iteration to set up overlap and histogram comparison
                matches = [(lapping, box, histDist)]
        
    if len(matches) > 0:
        #print(list(matches[0][1]),'selected boxes in bsroifinderbox')
        return list(matches[0][1])
    else:
        #print([],'selected boxes in bsroifinderbox')
        return []


    
    

    
    


def bsShapeExtractor(mask,roi):
   
    segmentStats= cv2.connectedComponentsWithStats(mask)
    
    if len(segmentStats[2])>1:
        segmentStats1 = segmentStats[2][1:]
        maxArea = np.max(segmentStats1[:,4],axis = 0)
        index = 0
        for i in segmentStats1[:,4]:
            if i == maxArea:
                break
            else:
                index = index+1
        mask[segmentStats[1]!=index+1] = [0] 
        return mask
    elif len(segmentStats[2])==1:
        #print(segmentStats[2])
        segmentStats1 = segmentStats[2][0]

        index = 0

        mask[segmentStats[1]!=index+1] = [0] 
        return []

def calculateDC(people, averageLocationList,previousDCList, homography, frame):
    DC = 0
    DCList = []
    currentFrameDC = []
    currentDC = []
    locations = np.array([],ndmin = 2)
    locations.shape = (0,2)
    i = 0
    for person in people:
        if len(person.locationArray)>=10:
            personArray = np.array(person.locationArray[-10:])
            locations = np.append(locations,personArray, axis = 0)
            i += 1
            
    #Just add code to grab the last ten locations for a person

    averageLocation = tuple([sum(y)/ len(y) for y in zip(*locations)])
    if len(averageLocation)> 0:
        px,py = worldToPixel(averageLocation,homography) #used to draw the circle
        cv2.circle(frame,((px,py)),5,(255,0,0),2)
    #print(averageLocation,'averageLocation in DC')
    averageLocationList.append(averageLocation)

    for location in locations:
        currentFrameDC = np.linalg.norm(location-averageLocation)
        #print(currentFrameDC,'currentFrame DC')
        DCList.append(currentFrameDC)
    DCList = np.array(DCList)
    currentDC = np.sum(DCList)
    previousDCList.append(currentDC)
    if len(previousDCList) > 1:
        pDC = np.array(previousDCList[-2])
        cDC = np.array(currentDC)
        #DC = np.linalg.norm(cDC - pDC)
        if pDC != 0.0:
            DC = np.divide(cDC,pDC)
    
    return DC,i       

#code for determining camera location, used when initializing simulator class
def cameraPosition(rotM,tvec):
    cameraPosition = -np.matrix(rotM).T * np.matrix(tvec) #perhaps pickle.dump this for future use    
    return cameraPosition
    
def dbScanAlgorithm(people):
    newPersonList = []
    newArray = np.array([],ndmin = 2)
    newArray.shape = (0,2)
    #sort the people first
    
    for person in people: #get people from person list
        if len(person.worldLocation) != 0:
            newPersonList.append(person)
    
    
    newPersonList.sort(key=lambda x: x.ID, reverse=False)  #sort the person list  
    
    for everyone in newPersonList:
        newArray = np.append(newArray,everyone.worldLocation, axis = 0)
    if len(newArray) != 0:
        
        db = DBSCAN(eps=4,min_samples=1).fit(newArray)
        core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print(labels,'labels') #need to iterate through the labels to assign a label to each person
        unique_labels = set(labels)
        i = 0
        for label in labels:
            person = newPersonList[i]
            person.clusterID = label
            i += 1
        return unique_labels
    else:
        return []




        
def dbScanAlgorithmROIs(rois,avgWidth):
    tmpArray = np.array([],ndmin = 2)
    tmpArray.shape = (0,2)
    returnROIs = []
    
    #sort the people first
    i = 0
    for box in rois: #get people from person list
        #box = list(box)
        #boxArray = np.array([box[0]+(box[2]/2),box[1]+box[3]],ndmin = 2) #bottom center of box
        boxArray = np.array([box[0]+(box[2]/2),box[1]+(box[3]/2)],ndmin = 2) #centroid of box
        boxArray.shape = (1,2)
        tmpArray = np.append(tmpArray,boxArray, axis = 0)
        rois[i] = list(rois[i])
        i += 1
    
    
    if len(tmpArray) != 0:
        
        db = DBSCAN(eps=avgWidth,min_samples=1).fit(tmpArray) #min_samples=1
        core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print(labels,'labels') #need to iterate through the labels to assign a label to each person
        unique_labels = set(labels)
        i = 0
        if len(labels) > 0:
            #currentlabel = labels[i]
            boxArray = np.array([],ndmin = 2)
            boxArray.shape = (0,2)
            rois2 = np.array(zip(rois,labels),ndmin = 2)
            for label in unique_labels:
                labeledRois = rois[rois2[:,1] == label] #filter(rois[:,1] == label, rois)
                #print(labeledRois,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                if len(labeledRois) == 1:
                    returnROIs.append(list(labeledRois[0]))
                elif len(labeledRois) > 1:
                    minX = np.min(labeledRois[:,0],axis = 0)
                    #print(minX)
                    minY = np.min(labeledRois[:,1],axis = 0)
                    widthSlice = labeledRois[:,0]+labeledRois[:,2]
                    maxX = np.max(widthSlice,axis = 0)
                    heightSlice = labeledRois[:,1]+labeledRois[:,3]
                    maxY = np.max(heightSlice,axis = 0)
                    box = [int(minX),int(minY),int(maxX-minX),int(maxY-minY)]
                   
                    returnROIs.append(box)
               
                     
               
                i += 1
    return returnROIs
    

    
def displayHistogram(histogram,frameNumber=-1,id=-1):
    histogram = histogram.reshape(-1)
    binCount = histogram.shape[0]
    BIN_WIDTH = 3
    img = np.zeros((256, binCount*BIN_WIDTH, 3), np.uint8)
    for i in xrange(binCount):
        h = int(histogram[i])
        cv2.rectangle(img, (i*BIN_WIDTH+1, 255), ((i+1)*BIN_WIDTH-1, 255-h), (int(180.0*i/binCount), 255, 255), -1) 
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if(frameNumber != -1):
        cv2.putText(img,'Frame#: %d' %frameNumber,(20,20),0, .75, (255,255,255), 1,8, False)
    if(id!=-1):
        cv2.imshow("Person "+str(id)+" Histogram", img)
    else:
        cv2.imshow("Probable Person Histogram", img)

def findAverageHeading(img,frameNumber,person,homography):
    index = len(person.kalmanLocation)
    #print(index,'index!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    if (index >= 20):
        points = np.array(person.kalmanLocation)
        pointA = [points[-20,0],points[-20,1]]
        points = points[-20:,0],points[-20:,1]
        points = np.array((points[0]-pointA[0],points[1]-pointA[1]))
        points = np.concatenate((points[:,:3],points[:,-8:]),axis = 1)
        points = np.transpose(points)
        vx, vy, cx, cy = cv2.fitLine((points), distType=1, param=0, reps=0.01 , aeps=0.01) 
        prvsPoint = (int(vx+pointA[0]), int(vy+pointA[1]))
        curPoint = (int(cx+vx+pointA[0]), int(cy+vy+pointA[1]))
        #cv2.line(img, prvsPoint, curPoint, (255, 0, 0),2)    
        prvsPoint = pixelToWorld(prvsPoint,homography)
        curPoint = pixelToWorld(curPoint,homography)
        fitLineHeading = findCurrentHeading(img,frameNumber,prvsPoint,curPoint,person.direction,person.fX,person.fY) #fix
        if fitLineHeading != -1:
            person.direction.append((frameNumber,fitLineHeading))  
        if len(person.direction) >= 20:
            headingList = []
            tmp = person.direction[:] 
            for i in range(20):
                headingList.append(tmp[-i][1])
            headingAvg = np.average(headingList)  
            
            #print('Avg %0.2f ' %headingAvg)
            #cv2.putText(img,'Avg %0.2f' %(headingAvg),(person.fX,person.fY),0,0.75,(0,0,255),1,8, False)
            return headingAvg
        else: 
            return -1
    else:
        #print('Not enough entries present to calculate heading')
        return -1
                
def findCurrentHeading(img,framenum,prvsPoint,curPoint,directionList,fX,fY):
    dx = curPoint[0] - prvsPoint[0]
    dy = curPoint[1] - prvsPoint[1]
    if dx > 0:
        heading = np.subtract(90,np.rad2deg(np.arctan((np.divide(dy,dx)))))#90
    elif dx < 0:
        heading = np.subtract(270,np.rad2deg(np.arctan((np.divide(dy,dx)))))#270
    else:
        if dy > 0:
            heading = 0
        elif dy < 0:
            heading = 180
        else: # dy == 0
            if directionList != []:
                heading = directionList[-1][1] #points are the same no change in bearing
            else:
                heading = -1
    if heading != -1:
        pass
        #print('fitLine heading %0.2f degrees.' % (heading))#angleBetaDeg))
        #cv2.putText(img,'Cur %0.2f' % (heading),(fX,fY-40),0, .75, (0,255,0), 1,8, False)   
    return(heading)


def findIntersections(A, B): #http://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    # min, max and all for arrays
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)
    aall = lambda abools: np.dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))

    x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = np.meshgrid(slope(A), slope(B))
    np.m1inv, m2inv = 1/m1, 1/m2

    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12), 
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    return xi[aall(xconds)], yi[aall(yconds)]



def foreGroundHist(img,fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM):
    mask = fgmask[fY:fY+fH,fX:fX+fW].copy()
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)  
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)#.7   
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    tempImg = img[fY:fY+fH,fX:fX+fW].copy()
    markers = cv2.watershed(tempImg,markers)
    mask[markers == -1] = [0]
    roiM = img[fY:(fY)+(fH), fX:(fX)+(fW)].copy() 
    
    
    hsv_roi =  cv2.cvtColor(roiM, cv2.COLOR_BGR2HSV)
    #===============================================
    #TEST_HSV_ROI=cv2.resize(hsv_roi,ROI_RESIZE_DIM)
    #cv2.imshow('hsv_roi',TEST_HSV_ROI)
    #===============================================
    
    
    mask1 = bsShapeExtractor(mask,[0,0,mask.shape[1],mask.shape[0]])
    #print(mask1)
    #==============================================
    #TEST_MASK_ROI=cv2.resize(mask,ROI_RESIZE_DIM)
    #cv2.imshow('segStatImg',TEST_MASK_ROI)
    #==============================================
    if len(mask1) > 0:
        mask = mask1
    #else:
        #return [] #in case where hist cannot be calculated
    
    
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    

    
    #Tools.displayHistogram(roi_hist,self.frameNumber)
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    del roiM
    del mask
    del mask1
    del tempImg
    return roi_hist

def roiFGBGHist(img,fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM): #not a foreground hist
    mask = fgmask[fY:fY+fH,fX:fX+fW].copy()

    roiM = img[fY:(fY)+(fH), fX:(fX)+(fW)].copy() 
    
    
    hsv_roi =  cv2.cvtColor(roiM, cv2.COLOR_BGR2HSV)

    
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    #Tools.displayHistogram(roi_hist,self.frameNumber)
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    del roiM
    del mask
    return roi_hist

def histogramComparison(curRoiHist,newRoiHist):
    distance = cv2.compareHist(curRoiHist,newRoiHist,4) #update based on color match 4
    return distance 
    
def kfAdjustment(person):
    if len(person.location) > 1:

        locationX = person.location[-1][1]
        locationY = person.location[-1][2]
        locLastX = person.kalmanX.prediction
        locLastY = person.kalmanY.prediction    
        dx = (locationX-locLastX)
        dy = (locationY-locLastY)
        person.kalmanX.step(dx)
        person.kalmanY.step(dy)

        if (person.kalmanX.covariance + person.kalmanY.covariance)/2 < .3: #.3        
            person.kalmanLocation.append((person.kalmanX.prediction + locLastX,person.kalmanY.prediction + locLastY))# ,  
        
def locationTracker(person,img):
    if len(person.location) > 1:
        locationList = []
        for i in person.location:
            locationList.append(i[1:])
        locationList = np.int0(locationList)
        kalmanLocation = np.int0(person.kalmanLocation)
        cv2.polylines(img,[locationList],False, (0,0,255),3) 
        cv2.polylines(img,[kalmanLocation],False, (255,0,0),6) #kalman line
        
def nonMaxSup(rects,thresh = 0.5):
    #print(rects)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=thresh) 
    i = 0
    for (xA, yA, xB, yB) in pick:
        pick[i][2] = xB-xA
        pick[i][3] = yB-yA
        i +=1
    return pick
    

def objectDistance(objectBottom,cameraPosition):
    bottom = np.array((objectBottom[0] ,objectBottom[1], 0))#or use z = 1 if trouble add one to camera height 
    cameraBase = np.array((cameraPosition[0],cameraPosition[1], 0))#or use z = 1
    cameraBaseToBottom = np.linalg.norm(cameraBase-bottom)#1 find distance from base of camera to bottom point
    return float(cameraBaseToBottom)
  
    
# code for finding an objects height in world coordinates
def objectHeight(bottomPoint,topPoint,cameraPosition):
    #1 find distance from base of camera to top point
    #2 find angle between line from base of camera to top toint, and line from top point to camera, assume camera is at 90deg angle from ground creating a right triangle.
    #3 find distance from bottom point to top point,
    #4 use the angle and the distance in step 3 to calculate height of object.
    cameraHeight = cameraPosition[2]
    bottom = np.array((bottomPoint[0] ,bottomPoint[1], 0))#or use z = 1 if trouble add one to camera height 
    top = np.array((topPoint[0], topPoint[1], 0))#or use z = 1
    cameraBase = np.array((cameraPosition[0],cameraPosition[1], 0))#or use z = 1
    cameraBaseToTop = np.linalg.norm(cameraBase-top)
    radAlpha = np.tan(np.divide(cameraHeight,cameraBaseToTop))
    bottomToTop = np.linalg.norm(bottom-top)
    tanAlpha = np.tan(radAlpha) #figure out why tan doesn't return expected values, has to do with degrees vs rad.
    objectHeight  =  np.multiply(bottomToTop, tanAlpha) #tanTheta)
    return float(objectHeight)
       

def overLap(a,b):  # returns 0 if rectangles don't intersect #a and b = [xmin ymin xmax ymax]  
    areaA = float((a[2]-a[0]) * (a[3]-a[1]))
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3],b[3]) - max(a[1],b[1])
    #print(dx,'dx')
    #print(dy,'dy')
    if (dx > 0 ) and (dy > 0):
        intersect = float(dx*dy)
        if areaA != 0:
            ratioA = intersect/areaA
        else:
            ratioA = 0
        return ratioA

    else:
        return 0        



def peopleMeanshift(img,imgCopy,people,width,height,term_crit,ROI_RESIZE_DIM):
    
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    track_window = people.fX, people.fY, people.fW, people.fH #track_window = 5, 5, people.fW, people.fH 
    #print(track_window)
    #dst = cv2.calcBackProject([hsv],[0],people.roiHist,[0,180],1) #[hsv[startY:endY,startX:endX]]
    dst = cv2.calcBackProject([hsv],[0],people.roiHist,[0,180],2) #[hsv[startY:endY,startX:endX]]
    #=========================================
    #TEST_DST_ROI=cv2.resize(dst,ROI_RESIZE_DIM)
    #cv2.imshow('dst',TEST_DST_ROI)
    #=========================================
    #cv2.imshow('dst',dst)
    cv2.rectangle(imgCopy, (people.fX, people.fY), (people.fX+people.fW, people.fY+people.fH), (255,0,0), 6) #show previous meanshift roi box blue
    # apply meanshift/camShift to get the new location
    ret, track_window = cv2.meanShift(dst,track_window,term_crit)
    x,y,w,h = track_window
    people.fX,people.fY = x,y  

    
# code for calculating world coordinates from pixel coordinates
def pixelToWorld(imagePoint,homography):
    if not homography == []:
        pixelVector = ([[imagePoint[0]],[imagePoint[1]],[1]])
        pixelVector = np.array(pixelVector,dtype=np.float32)
        worldVector = np.dot(homography, pixelVector)
        worldVector = np.divide(worldVector,worldVector[2])
        return (worldVector[0], worldVector[1])
    else:
        print("You must first create homography matrix.")  
        

def segmentVisualization(segmentStats,fh,fw):
    labelMatrix = segmentStats[1]
    img = np.zeros((300,200,3), np.uint8)
    fx=0
    fy=0
    while fy< fh:
        fx=0
        while fx<fw:
            if labelMatrix[fx,fy]==1:
                cv2.circle(img,(fy,fx),2,(0,0,255),1)
            elif labelMatrix[fx,fy]==2:
                cv2.circle(img,(fy,fx),2,(0,255,0),1)
            elif labelMatrix[fx,fy]==3:
                cv2.circle(img,(fy,fx),2,(255,0,0),1)
            fx=fx+1
        fy=fy+1
        cv2.imshow("Connected Components",img)




def thresholdRegionExtractor(fgmask):
    # Matrix Structure for segmentStats:
    # ----------------------------------
    # Index           Description
    # ===========================================================================
    # 0               Total number of connected segments
    # 1               Matrix mapping of connected components
    # 2               Array of lists containing bounding box and area [x,y,w,h,a]
    # 3               Array of lists of centroids
    
    segmentStats= cv2.connectedComponentsWithStats(fgmask,connectivity = 4) # connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]
    if segmentStats[0]!=0:
        if len(segmentStats[2])>1:
            segmentStats = segmentStats[2][1:]
            #roi = segmentStats[segmentStats[:,4]>200]
            roi = segmentStats[segmentStats[:,4]>100]

            roiAvgWidth = 80
            #print(roiAvgWidth,'avg width in region extractor')
            roi = roi[:,:4]

            roi = dbScanAlgorithmROIs(roi,roiAvgWidth)
            
            roi = nonMaxSup(roi,0.5) #.3
            
            return roi
        else:
            return []
    else:
        return []

        

def worldToPixel(worldPoint,homography): 
    if not homography == []:
        worldVector = np.array([[worldPoint[0]], [worldPoint[1]], [1]], dtype= 'float64')
        invHomography = np.linalg.inv(homography)
        pixelVector = np.abs(np.dot(invHomography,worldVector))
        pixelVector = np.divide(pixelVector,pixelVector[2]).astype(int)
        return pixelVector[0], pixelVector[1]
    else:
        print("You must first create homography matrix.")
    

    
class KalmanFilter():
#Credit to Nicholas Kennedy. Renaming function for simplicity's sake.

    def __init__(self,prediction,kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0):
        self.kalmanGain = 0
        self.covariance = 1.0 #1.0
        self.measurmentNoiseModel = 1.5#.8
        self.prediction = prediction
        self.covarianceGain = 1.10#1.05
        self.lastSensorValue = 0
        
    def updatePrediction(self, sensorValue = None):
        if sensorValue != None:
            self.prediction = self.prediction+self.kalmanGain*(sensorValue-self.prediction)
        else:
            self.prediction = self.prediction+self.kalmanGain*(0-self.prediction)
        #return self.prediction
    
    def updateCovariance(self):
        self.covariance = (1-self.kalmanGain)*self.covariance#(1-self.kalmanGain)*self.covariance
        #print(self.covariance,'covariance')
    
    def updateKalmanGain(self):
        self.kalmanGain = (self.covariance)/(self.covariance+self.measurmentNoiseModel)#(1+self.covariance)/(self.covariance+self.measurmentNoiseModel)
        
    def step(self, sensorValue = None):
        self.covariance = self.covariance * self.covarianceGain
        self.updateKalmanGain()
        self.updatePrediction(sensorValue)
        
        if sensorValue != None:
            self.updateCovariance()
            self.lastSensorValue = sensorValue
        else:
            self.covariance = (1-self.kalmanGain)*self.covariance#(self.kalmanGain)*self.covariance#(1-self.kalmanGain)*self.covariance
        