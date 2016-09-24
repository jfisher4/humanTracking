import cv2
import numpy as np
import os
import pickle 
import Tools
import time
import MovementPatterns
import random
#Global Variables
hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05,'finalThreshold': 1}
ROI_RESIZE_DIM = (64,128)
#WINDOW_RESIZE_DIM = (845,475)
WINDOW_RESIZE_DIM = (1440,810)
timeStart = time.time()
  
class Simulator:

    def __init__(self, directory, videoname):
        self.directory = directory
        os.chdir(directory)
        self.cap = cv2.VideoCapture(videoname)
        self.metadata = videoname.split("_")  
        self.homography = pickle.load( open( self.metadata[0]+"_H.p", "rb" ) )
        self.rotationMatrix = pickle.load(open( self.metadata[0]+"_rotM.p","rb"))
        self.cameraPosition = Tools.cameraPosition(self.rotationMatrix, pickle.load( open( self.metadata[0]+"_tvec.p", "rb" ) ))
        self.frameNumber = 0
        self.HOGClassifier = cv2.HOGDescriptor()
        self.HOGClassifier.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.trackedPeople = People()
        _, self.prvs = self.cap.read()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history = 0, varThreshold =64, detectShadows = False)
        self.fgbg.apply(self.prvs)
        self.height = int(self.prvs.shape[0])
        self.width = int(self.prvs.shape[1])
        self.bSROIs= []
        self.termCrit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1)
        self.fgmask = []
        self.stateInspector = MovementPatterns.CrowdBehaviorDetector()
        self.currentLabels = []
        self.movementReport = []
        self.groupROIs = []
        self.kernel = Tools.np.ones((5,5),Tools.np.uint8)
        self.clusteredGroups = []
        self.averageLocationList = []
        self.previousDCList = []
        self.lengthValidPeople = 0
        self.waitingList = []
        self.dcKalman = None
        global HEIGHT,WIDTH        
        HEIGHT = self.height
        WIDTH = self.width
        
        
    def retrieve(self):
        time1 = time.time()
        ret,img = self.cap.read()
        if not ret: #allow for a graceful exit when the video ends
            print("Exiting Program End of Video...")
            self.cap.release()
            Tools.cv2.destroyAllWindows()
            return(None, 0) #return 0 to toggle active off
        self.frameNumber += 1
        print('framenumber ' + str(self.frameNumber))
        #print('W', WIDTH, 'H', HEIGHT)
        imgDisplay = img.copy()
        self.fgmask=self.fgbg.apply(img)
        roi = Tools.thresholdRegionExtractor(self.fgmask)
        time2 = time.time()
        print('time for conncomp,dbscan and nonmax ',time2-time1)
        #print(roi)
        unusedROIs = [] #list(roi)
        self.bSROIs= []
        filteredHOGResult = []
        if len(roi) > 20:
            return (None,1)
        
        
        for r in roi:
            bottomPoint = Tools.pixelToWorld((r[0]+(r[2]/2),r[1]+r[3]), self.homography)
            topPoint = Tools.pixelToWorld((r[0]+(r[2]/2),r[1]), self.homography)
            tmpHeight = Tools.objectHeight(bottomPoint, topPoint, self.cameraPosition)
            leftPoint = Tools.pixelToWorld((r[0],r[1]+r[3]), self.homography)
            rightPoint = Tools.pixelToWorld((r[0]+r[2],r[1]+r[3]), self.homography)
            tmpWidth = Tools.objectDistance(leftPoint,rightPoint)
            #print(tmpHeight,'tmpheight')
            boxA = r[2]*r[3]
            imgA = self.width*self.height
            ratioA = float(boxA)/float(imgA)
            #print(ratioA,'ratioA')
            cv2.rectangle(imgDisplay, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (128,128,0), 4)
            if tmpHeight > 1.25 and tmpHeight < 3.5 and tmpWidth > .3  and tmpWidth < 6 and ratioA < .15:# and tmpWidth < 2 and ratioA < .15:
                self.bSROIs.append(r)
                
                cv2.rectangle(imgDisplay, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (128,0,128), 8)
                
                

        
        for person in self.trackedPeople.listOfPeople: #update all peoples roiCurrent, and let hog search for them in their roiCurrent if visibility value is over 30.
            if len(person.roiCurrent) > 0:
                person.lastROICurrent = person.roiCurrent
             
            box = Tools.bsRoiFinderPerson(person,self.bSROIs,self.fgmask,img,ROI_RESIZE_DIM) #
            if len(box) > 0:
                person.roiCurrent = box
            else:
                person.roiCurrent = []
            
            if person.V > 30 and person not in self.waitingList: 
                self.waitingList.append(person)

        
        #print(self.bSROIs,'bSROIs')
        for box in self.bSROIs:
            count = 0
            for person in self.trackedPeople.listOfPeople:
                if any(map(lambda x: x in box, person.roiCurrent)):
                    count += 1
            if count == 0:
                unusedROIs.append(box)
        
        #print(unusedROIs,'unusedROIs')
        if len(unusedROIs)> 0: # chose 1 unused BS ROI to scan with HOG
            
            box = random.choice(unusedROIs)
            startX = box[0]-50          
            endX = box[0]+box[2]+50     
            startY = box[1]-50
            endY = box[1]+box[3]+50
            if startX < 0:           
                startX = 0
            if endX >= self.width-1:
                endX = self.width
            if startY < 0:
                startY = 0
            if endY >= self.height-1:
                endY = self.height    
            if endY > 1:                                   #omit boxes that are in skyline and have an area of 1/4 of the screen size
                result = self.HOGClassifier.detectMultiScale(img[startY:endY,startX:endX], **hogParams)
                #print(result)
                if result != ((),()):
                    result = result[0]
                    for r1 in result:
                        r1[0] = r1[0] + startX
                        r1[1] = r1[1] + startY
                        filteredHOGResult.append(r1)
        #print(self.waitingList,'waitingList')               
        elif len(self.waitingList)> 0: #or try to find a previous person that hasnt been detected in over 30 frames
            person = self.waitingList[0]
            startX = person.fX-50          
            endX = person.fX+person.fW+50     
            startY = person.fY-50
            endY = person.fY+person.fH+50
            if startX < 0:           
                startX = 0
            if endX >= self.width-1:
                endX = self.width
            if startY < 0:
                startY = 0
            if endY >= self.height-1:
                endY = self.height    
            if endY > 1:                                   #omit boxes that are in skyline and have an area of 1/4 of the screen size
                result = self.HOGClassifier.detectMultiScale(img[startY:endY,startX:endX], **hogParams)
                #print(result)
                if result != ((),()):
                    result = result[0]
                    #print(result,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',person.ID)
                    person.V = 0
                    person.roiCounter = 0
                    self.waitingList.pop(0)
            
        
        if len(self.bSROIs)< len(self.trackedPeople.listOfPeople):
            value = self.fgbg.getHistory()
            self.fgbg.setHistory(value+10)
            
        else:
            value = self.fgbg.getHistory()
            if value > 1:
                self.fgbg.setHistory(value - 1)
        
        filteredHOGResult = Tools.nonMaxSup(filteredHOGResult,0.1) #call Non maxima supression to reduce the overlapping boxes
        
        
        if filteredHOGResult != []:
            #print('condition1')
            #print(filteredHOGResult,'newhog result2')      
            for r in filteredHOGResult:
                fX, fY, fW, fH = r
                bottomPoint = Tools.pixelToWorld((fX+(fW/2),fY+fH), self.homography)
                topPoint = Tools.pixelToWorld((fX+(fW/2),fY), self.homography)
                tmpHeight = Tools.objectHeight(bottomPoint, topPoint, self.cameraPosition)
                if tmpHeight > 1.75 and tmpHeight < 5:
                    cv2.rectangle(imgDisplay, (fX, fY), (fX+fW, fY+fH), (0,128,255), 6)
                    
                    roi_hist = Tools.roiFGBGHist(img,self.fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM)
                    
                    #Tools.displayHistogram(roi_hist,self.frameNumber)
                    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                    self.trackedPeople.update(img,self.fgmask,fX,fY,fW,fH,self.frameNumber,roi_hist,tmpHeight,self.bSROIs,self.homography,self.cameraPosition,self.trackedPeople.listOfPeople)
                    
        time3 = time.time()
        print('time to process rois and scan with hog ',time3-time2)
                    
        self.trackedPeople.refresh(img,imgDisplay,self.fgmask,self.frameNumber,self.bSROIs,self.homography,self.cameraPosition,self.termCrit,self.waitingList)                 
        #print('len(tracked people)',len(self.trackedPeople.listOfPeople))
        time4 = time.time()
        print('time to refresh people ',time4-time3)
        #movement pattern code, every 30 frames
        if self.frameNumber % 1 == 0 and  len(self.trackedPeople.listOfPeople) > 0:
            labels = [0]
            for clusterLabel in labels:
                #print(clusterLabel,'clusterlabel')
                if clusterLabel != -1:
                    if clusterLabel in self.currentLabels:
                        cluster = filter(lambda x: x.label == clusterLabel, self.clusteredGroups)[0]
                        #print('success in getting clusteredGroup')
                    else:
                        #print('failed to get clusteredGroup')
                        cluster = ClusterGroup(clusterLabel)
                        self.clusteredGroups.append(cluster)
                    #print(cluster.label,'label ##############################################')
                    peopleList = []
                    for person in self.trackedPeople.listOfPeople: #get people from regular list
                        if person.clusterID == clusterLabel:
                            peopleList.append(person)
                    
                    cluster.people = peopleList #put people in the clusterGroup
                    cluster.previousLen = len(cluster.people)
                    cluster.state = 1
                    for person in cluster.people: #put clusterGroup in people
                        person.clusterGroup = cluster
                    
            for clusterGroup in self.clusteredGroups: #update cluster states for next iteration and remove inactive clusters
                if clusterGroup.state == 1: # if 1 then clusterGroup is active, so set to zero
                    clusterGroup.state = 0
                   
                else: # else clusterGroup is inactive so remove
                    self.clusteredGroups.remove(clusterGroup)
                    del clusterGroup
                    #print('removed')
                
                
            self.currentLabels = labels
            #print ("Here are the labels: ",labels)
        dc = 0.0
        if len(self.trackedPeople.listOfPeople) > 1:
            
            DC,newLengthValidPeople = Tools.calculateDC(self.trackedPeople.listOfPeople, self.averageLocationList,self.previousDCList, self.homography,imgDisplay)
            if self.dcKalman is None:
                self.dcKalman = Tools.KalmanFilter(DC,kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0)
            else:
                if self.lengthValidPeople == newLengthValidPeople:
                    self.dcKalman.step(DC)
            self.lengthValidPeople = newLengthValidPeople
            dc = self.dcKalman.prediction
            #print(DC,'DC!!!!!!!!!!!!!!!!!!!!!!!!##############################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            #print(self.dcKalman.prediction,'dcKalman.prediction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@&&&&&&&&&&&&&&&&&&&&&&&')
        self.movementReport = self.stateInspector.updateState(self.trackedPeople.listOfPeople,self.clusteredGroups,dc)
        #print(self.movementReport,'movementreport')            
        for person in self.trackedPeople.listOfPeople:
            if person.V == 1:# HOG has updated visibility this frame
                
                cv2.rectangle(imgDisplay, (person.fX, person.fY), (person.fX+person.fW,person.fY+person.fH), (0,0,255), 2)
                cv2.putText(imgDisplay,str(person.ID),(person.fX+5,person.fY+30),0, 1, (0,0,255), 3,8, False)
            elif person.V < 1000: #HOG did not update visibility and person was tracked with background subtracction
                cv2.rectangle(imgDisplay, (person.fX, person.fY), (person.fX+person.fW, person.fY+person.fH), (0,255,0), 4) #show meanshift roi box green
                cv2.putText(imgDisplay,str(person.ID),(person.fX+5,person.fY+30),0, 1, (0,255,0), 3,8, False)
        

            person.heading = Tools.findAverageHeading(imgDisplay,self.frameNumber,person,self.homography)
            #print(person.heading,'test heading')
            Tools.kfAdjustment(person) 
            if person.heading != -1:
                Tools.cv2.putText(imgDisplay,'Avg %0.2f' %(person.heading),(person.fX,person.fY),0,0.75,(0,0,255),1,8, False)
            Tools.cv2.putText(imgDisplay,str(person.clusterID),(person.fX+(person.fW/2)-20,person.fY+(person.fH/2)),0, 2, (0,128,255), 3,8, False)
            if len(person.locationArray)> 30 and self.frameNumber % 1 == 0: # find speed every 30 sec
                previousPoint = [person.locationArray[-30][0],person.locationArray[-30][1]]
                currentPoint = [person.locationArray[-1][0],person.locationArray[-1][1]]
                distance = Tools.objectDistance(previousPoint,currentPoint)
                #print(distance,'speed')
                if person.speed != -1:
                    person.speed = Tools.np.average(Tools.np.array([person.speed,distance]))
                else:
                    person.speed = distance
                #print(person.speed,'speed')
                if person.speed < .5: #person is not moving
                    person.moving = False
                    person.running = False
                elif person.speed >= .5 and person.speed <2.5: #person is moving slowly
                    person.moving = True
                    person.running = False
                else: #person is moving fast
                    person.moving = True
                    person.running = True

        if len(self.movementReport) <= 3:            
            Tools.cv2.putText(imgDisplay,str(self.movementReport),(250,800),0, 1, (0,128,255), 3,8, False)
        else:
            reportText = ""
            textY = 800
            for index in range(len(self.movementReport)):
                reportText = reportText + str(self.movementReport[index])
                if index % 3 == 0:
                    Tools.cv2.putText(imgDisplay,str(reportText),(250,textY),0, 1, (0,128,255), 3,8, False)
                    reportText = ""
                    textY += 40
                elif index == len(self.movementReport)-1:
                    Tools.cv2.putText(imgDisplay,str(reportText),(250,textY),0, 1, (0,128,255), 3,8, False)
        
        time5 = time.time()
        print('time to do clustering and movement report ',time5-time4)
        
        
        # Recycling code
        fgMaskDisplay = cv2.resize(self.fgmask, WINDOW_RESIZE_DIM)
        cv2.imshow('BSub',fgMaskDisplay)
        time6 = time.time()
        print('total time ',time6-time1)
        framePerSec = 1/(time6-time1)
        printString = str(framePerSec)+' Frames per second'
        cv2.putText(imgDisplay,printString,(20,40),0,1, (0,0,255),3,8,False)
        printString = 'Frame ' + str(self.frameNumber)
        cv2.putText(imgDisplay,printString,(20,80),0,1, (0,0,255),3,8,False)
        imgDisplay = cv2.resize(imgDisplay, WINDOW_RESIZE_DIM)
        cv2.imshow(self.metadata[0],imgDisplay)
        #if self.metadata[0][8] =="A":
            #cv2.moveWindow(self.metadata[0],0,0)
        #else:
            #cv2.moveWindow(self.metadata[0],770,360)
        print('\n')
        k = Tools.cv2.waitKey(2) & 0xFF 
        if k == ord('p'):
            print("Pausing...")
            return (None,2) #return 2 for paused
        elif k == ord('q'):
            print("Exiting Program...")
            self.cap.release()
            Tools.cv2.destroyAllWindows()
            return (None,0) #return 0 to toggle active off
        if self.frameNumber == 400: #for testing only to pause at a certain frame
            timeEnd = time.time()
            totalTime = timeEnd - timeStart
            print(totalTime,'totalTime')
            
        elif self.frameNumber == 401: #for testing only to pause at a certain frame
            
            return (None,2)
        return (None,1) #return 1 to stay active

class People():
    ## The constructor.
    def __init__(self):
        self.listOfPeople=list()
        self.lostListOfPeople=list()
        self.index=0

# Updates an item in the list of people/object or appends a new entry or assigns to a group or removes from a group      
    def update(self,img,fgmask,fX,fY,fW,fH,frameNumber,roiHist,height,bSROIs,homography,cameraPosition,listOfPeople):
        
        matches = []
        #lostFlag = 0
        occlCandidate = [] #list to hold a group of people that a person may be added to
        i = 0
        #print[fX,fY,fW,fH]
        box0 = Tools.bsRoiFinderBox([fX,fY,fW,fH],roiHist,bSROIs,fgmask,img,ROI_RESIZE_DIM)#corresponding roi for hog box
        box2 = [fX,fY,fX+fW,fY+fH]
        p1 = Tools.pixelToWorld((fX+(fW/2),fY+fH), homography)
        if len(box0) == 0:
            return
        else:
            if box0[0] <= 100 or box0[0]+box0[2] >= WIDTH-100 or box0[1] <= 2 or box0[1]+box0[3] >= HEIGHT-20 : #check if person1 is on edge of scene
                boxOnEdge = True
                
            else:
                boxOnEdge = False

        
        if len(matches) == 0: #new method 1
            i = 0
            for person in self.listOfPeople:
                box1 = [person.fX, person.fY, person.fX+person.fW, person.fY+person.fH]
                lapping = Tools.overLap(box1,box2) #largest overlap
                if lapping > 0:
                    histDist = Tools.histogramComparison(roiHist,person.roiHist)
                    if len(matches)>0:
                        if lapping >= matches[0][0]:
                            if histDist < matches[0][3]:
                                matches = [(lapping, i,0,histDist)]  #flag of one means it was found in  lost people
                    else: #used in first iteration to set up overlap and histogram comparison
                        matches = [(lapping, i,0,histDist)]
                i = i + 1
        
        
        
        if len(matches) == 0 and boxOnEdge == False: #try to assign to person that is sharing a ROI
            i = 0
            for person in self.listOfPeople:
                if person.sharedROI == True:
                    p2 = Tools.pixelToWorld((person.fX+(person.fW/2),person.fY+person.fH), homography)
                    dist = Tools.objectDistance(p1,p2)
                    if dist < 5:
                        histDist = Tools.histogramComparison(roiHist,person.roiHist)
                    #print(histDist,'histDist 323')
                        if len(matches)>0: #used after first iteration to compare overlap and histogram 
                        #if lapping > matches[0][0]:
                            if histDist < matches[0][3]:
                                matches = [([], i,0,histDist)]  #flag of one means it was found in  lost people
                        else: #used in first iteration to set up overlap and histogram comparison
                            matches = [([], i,0,histDist)]
                i = i + 1
            if len(matches) > 0:
                
                person = self.listOfPeople[matches[0][1]]
                if person.V > 0:
                    person.roiCurrent = box0
                    person.lastGoodROI = box0
                    person.lastROICurrent = box0
                    person.fX,person.fY,person.fW,person.fH = person.roiCurrent[0],person.roiCurrent[1],person.roiCurrent[2],person.roiCurrent[3]
                    person.V=0
                    person.edgeCounter = 0
                    person.roiCounter = 0
                    #print('distance is '+str(matches[0][0])+' '+ 'match found for ' +str(person.ID)+' in update case 0',lostFlag,'lostFlag')
                return
                
        
        if len(matches) == 0: #check lost people for match
            i = 0
            for person in self.lostListOfPeople:
                histDist = Tools.histogramComparison(roiHist,person.roiHist)
               # print(histDist,'histDist 323')
                if len(matches)>0: #used after first iteration to compare overlap and histogram 
                    #if lapping > matches[0][0]:
                    if histDist < matches[0][3]:
                        matches = [([], i,1,histDist)]  #flag of one means it was found in  lost people
                else: #used in first iteration to set up overlap and histogram comparison
                    matches = [([], i,1,histDist)]
                i = i + 1
            if len(matches) > 0:
                if matches[0][3]> 7000: #hard coded value based on observations
                    
                    matches = [] #histogram match is not close enough make  a new person
                    #lostFlag = 0
                else:
                    person = self.lostListOfPeople[matches[0][1]]
                    pointA = Tools.pixelToWorld((person.location[-1][1],person.location[-1][2]),homography)
                    pointB = Tools.pixelToWorld((fX+(fW/2),fY+(fH/2)),homography)
                    distM = Tools.objectDistance(pointA,pointB)
                    if distM < 5:
                        pass
                        #lostFlag = 1
                    else:
                        matches = []
                        #lostFlag = 0

        
        if len(matches)>0 and len(occlCandidate) == 0: #1 match found and no occlusion update person attributes update person
            flag = matches[0][2]
            index = matches[0][1]
            if flag == 0: #get the person from matches
                person = self.listOfPeople[index]
            else:
                person = self.lostListOfPeople[index]
                self.insertPerson(person,self.listOfPeople)
                self.removePerson(person.ID,self.lostListOfPeople)
             
            if person.V > 0:#frameNumber > person.location[-1][0]: #if this is the first hog box for a person in this frame update person
                if len(person.roiHistList)< 5: #try to optimize the persons color histogram
                    
                    p1 = Tools.pixelToWorld((fX+(fW/2),fY+fH),homography)
                    distance = -1
                    for person2 in self.listOfPeople:
                        if person2.ID != person.ID:
                            p2 = Tools.pixelToWorld((person2.fX + (person2.fW/2),person2.fY+person2.fH),homography)
                            distance2 = Tools.objectDistance(p1,p2)
                            #distance.append(distance2)
                            #print('distance in hist update', distance2)
                            if distance != -1:
                                if distance2 < distance:
                                    distance = distance2
                                else:
                                    pass
                            else:
                                distance = distance2
                               # print('test')
                        
                   # print('distance in hist update', distance)
                    #print(min(distance),'min')
                    if distance == -1 or distance > 1:
                        roiHist2 = Tools.roiFGBGHist(img,fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM)
                        person.roiHistList.append(roiHist2)
                        #Tools.displayHistogram(person.roiHist,frameNumber,person.ID)
                    if len(person.roiHistList) > 1: #optimize the histogram
                    
                        #print(person.roiHistList)                     
                        for hist in person.roiHistList:
                            for i in range(len(person.roiHist)):
                                person.roiHist[i] = person.roiHist[i]+hist[i]
                        for i in range(len(person.roiHist)):
                            person.roiHist[i] = person.roiHist[i]/len(person.roiHistList)
         
                    
                person.V=0
                person.edgeCounter = 0
                person.roiCounter = 0
                #print('distance is '+str(matches[0][0])+' '+ 'match found for ' +str(person.ID)+' in update case 1',lostFlag,'lostFlag')
                return
            else:
                
                return


        
        
        elif len(matches) == 0 and len(occlCandidate) == 0:#4 no match found so create person after refining the hog box
            roiHist = Tools.foreGroundHist(img,fgmask,fX,fY,fW,fH,ROI_RESIZE_DIM)
            tmp_node=Person(self.index,fX,fY,fW,fH,0,roiHist,height) #step 3 only update persons histogram on creation, not in subsequent updates.
            #Tools.dbScanAlgorithmRoiCompare(bSROIs,[tmp_node],fgmask,img,ROI_RESIZE_DIM)
            tmp_node.roiCurrent = Tools.bsRoiFinderPerson(tmp_node,bSROIs,fgmask,img,ROI_RESIZE_DIM)
            if len(tmp_node.roiCurrent) > 0:
                tmp_node.fX,tmp_node.fY,tmp_node.fW,tmp_node.fH = tmp_node.roiCurrent[0],tmp_node.roiCurrent[1],tmp_node.roiCurrent[2],tmp_node.roiCurrent[3]
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                return
            else:
                tmp_node.roiCurrent = [fX,fY,fW,fH]
                #tmp_node.roiCurrent = []
                self.listOfPeople.append(tmp_node)
                self.index=self.index+1
                #print('new person added roiCurrent cheated, index is '+ str(self.index) +' case 4e')
                return
                    
    def refresh(self,img,imgCopy,fgmask,frameNumber,bSROIs,homography,cameraPosition,term_crit,waitingList): #updates people's boxes and checks for occlusion
        personList = list(self.listOfPeople) #make copy of people list to use for while loop
        
        while len(personList) > 0:
            
            person1 = self.getPerson(personList[0].ID,self.listOfPeople)
            #print(person1.ID,'moving = ', person1.moving)
            flag = 0
            person1.V=person1.V+1
            if len(person1.roiCurrent) != 0:
                person1Box2 = [person1.roiCurrent[0], person1.roiCurrent[1], person1.roiCurrent[2], person1.roiCurrent[3]]
            else:
                person1Box2 = [person1.lastROICurrent[0], person1.lastROICurrent[1], person1.lastROICurrent[2], person1.lastROICurrent[3]]
            for person2 in self.listOfPeople: #determine whether person shares BSroi with other person and add to new group if so
                if person1.ID != person2.ID:
                    if len(person2.roiCurrent) != 0:
                        person2Box = [person2.roiCurrent[0], person2.roiCurrent[1], person2.roiCurrent[2], person2.roiCurrent[3]]
                    else:
                        person2Box = [person2.lastROICurrent[0], person2.lastROICurrent[1], person2.lastROICurrent[2], person2.lastROICurrent[3]]
                        
                    if person1Box2 == person2Box:
                        flag = 1
                              
                                
            #print(person1.ID, flag, 'flag for current person')                

            if flag == 0 and len(person1.roiCurrent) > 0 and person1.V == 1:
                
                bottomPoint = Tools.pixelToWorld((person1.fX+(person1.fW/2),person1.fY+person1.fH), homography) #find location of bottom center of the roi
                topPoint = Tools.pixelToWorld((person1.fX+(person1.fW/2),person1.fY), homography)         #find location of the top center of the roi
                tmpHeight = Tools.objectHeight(bottomPoint, topPoint, cameraPosition)   #calculate the height of the roi
                #print(box,'BSFinderbox 114')
                #print(person.location)
                if tmpHeight >= 1.25: #filter out bad regions based on height
                    
                    #find avg height for person
                    if len(person1.heightList) < 35:
                        #if person.V == 1:
                        person1.heightList.append(tmpHeight)
                        
                    else:
                        if person1.height == -1:
                            person1.height = np.average(person1.heightList)
                            
                    leftPoint = Tools.pixelToWorld((person1.fX,person1.fY+person1.fH), homography)
                    rightPoint = Tools.pixelToWorld((person1.fX+person1.fW,person1.fY+person1.fH), homography)
                    tmpWidth = Tools.objectDistance(leftPoint,rightPoint)        
                    if len(person1.widthList) < 35:
                        person1.widthList.append(tmpWidth)
                        
                    else:
                        if person1.width == -1:
                            person1.width = np.average(person1.widthList)    
            
      
            
            if len(person1.roiCurrent) == 0 and person1.nearEdge == True and person1.edgeCounter > 15:# and person1.meanShiftStateCounter > 120: #code to detect the person leaving the scene
                if person1 in waitingList:
                    waitingList.remove(person1)
                self.insertPerson(person1,self.lostListOfPeople)
                #print(person1.ID,'sent to lost people left edge of scene')
                self.removePerson(person1.ID,self.listOfPeople)
                personList.remove(person1)
                continue #skip to next person            

            if len(person1.roiCurrent) == 0  and person1.roiCounter > 1000 and person1.V > 120:# and person1.meanShiftStateCounter > 120: #code to detect the person leaving the scene
                if person1 in waitingList:
                    waitingList.remove(person1)
                self.insertPerson(person1,self.lostListOfPeople)
                #print(person1.ID,'sent to lost people lost in scene')
                self.removePerson(person1.ID,self.listOfPeople)
                personList.remove(person1)
                continue #skip to next person
            
            if len(person1.roiCurrent) > 0 and flag == 1: #ROI is shared
                person1.sharedROI = True
                if person1.moving == True:
                    person1.lastGoodROI = person1.roiCurrent
                    #print('case1a in refresh, meanshift on last location, roi is shared')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6)
   
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                    
                    if person1.fX < person1.roiCurrent[0]: # code to keep box from wondering
                        person1.fX = person1.roiCurrent[0]
                        #print('fixed box location 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > person1.roiCurrent[0]+person1.roiCurrent[2]:
                        diff = (person1.fX+person1.fW) - (person1.roiCurrent[0]+person1.roiCurrent[2])
                        a = Tools.np.array((person1.fX+person1.fW))
                        b = Tools.np.array((person1.roiCurrent[0]+person1.roiCurrent[2]))
                        dist = Tools.np.linalg.norm(a-b)
                        
                        print(diff)
                        print(dist)
                        if person1.fX > person1.roiCurrent[0]+diff:
                            person1.fX = person1.fX - diff
                            
                            #print('fixed box location 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        elif person1.roiCurrent[0] == 0: 
                            person1.fX = person1.roiCurrent[0]
                    
                    if person1.fY < person1.roiCurrent[1]: # code to keep box from wondering
                        person1.fY = person1.roiCurrent[1]
                        #print('fixed box location 3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > person1.roiCurrent[1]+person1.roiCurrent[3]:
                        diff = (person1.fY+person1.fH) - (person1.roiCurrent[1]+person1.roiCurrent[3])
                        a = Tools.np.array((person1.fY+person1.fH))
                        b = Tools.np.array((person1.roiCurrent[1]+person1.roiCurrent[3]))
                        dist = Tools.np.linalg.norm(a-b)
                        
                        if person1.fY > person1.roiCurrent[1]+diff:
                            #person1.fY = person1.roiCurrent[1]
                            person1.fY = person1.fY - diff
                            #print('fixed box location 4 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        
                elif person1.moving == False:
                    #print('case1b in refresh, meanshift on last location, roi is shared')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6)
   
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                    
                    if person1.fX < person1.roiCurrent[0]: # code to keep box from wondering
                        person1.fX = person1.roiCurrent[0]
                        #print('fixed box location 1b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > person1.roiCurrent[0]+person1.roiCurrent[2]:
                        diff = (person1.fX+person1.fW) - (person1.roiCurrent[0]+person1.roiCurrent[2])
                        if person1.fX > person1.roiCurrent[0]+diff:
                            person1.fX = person1.fX - diff
                            #print('fixed box location 2b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                    if person1.fY < person1.roiCurrent[1]: # code to keep box from wondering
                        person1.fY = person1.roiCurrent[1]
                        #print('fixed box location 3b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > person1.roiCurrent[1]+person1.roiCurrent[3]:
                        diff = (person1.fY+person1.fH) - (person1.roiCurrent[1]+person1.roiCurrent[3])
                        if person1.fY > person1.roiCurrent[1]+diff:
                            person1.fY = person1.fY - diff
                            #print('fixed box location 4b !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                
            
            elif len(person1.roiCurrent) == 0 and flag == 0:# : # do for every person with no BS roi
                
                person1.roiCounter += 1
                if person1.edgeCounter == 0 and person1.moving == True:
                    #print('case2a in refresh, no current ROI, adjust box and meanshift')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6) 
                    previousFX = person1.lastGoodROI[0]
                    previousFY = person1.lastGoodROI[1]
                    previousFW = person1.lastGoodROI[2]
                    previousFH = person1.lastGoodROI[3]
                    #person1.fX,person1.fY,person1.fW,person1.fH = person1.lastGoodROI[0],person1.lastGoodROI[1],person1.lastGoodROI[2],person1.lastGoodROI[3]
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                    
                    if person1.fX < previousFX: # code to keep box from wondering
                        person1.fX = previousFX
                        #print('fixed box location 5 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > previousFX+previousFW :
                        diff = (person1.fX+person1.fW) - (previousFX+previousFW)
                        if person1.fX > previousFX + diff:
                            #person1.fX = person1.roiCurrent[0]
                            person1.fX = person1.fX - diff
                            #person1.fX = int(previousFX - .2*previousFW)
                            #print('fixed box location 6 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                    if person1.fY < previousFY: # code to keep box from wondering
                        person1.fY = previousFY
                        #print('fixed box location 7 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > previousFY+previousFH:
                        diff = (person1.fY+person1.fH) - (previousFY+previousFH)
                        if person1.fY > previousFY + diff:
                            person1.fY = person1.fY - diff
                        #person1.fY = int(previousFY - .1*previousFH)
                            #print('fixed box location 8 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                
                elif person1.edgeCounter == 0 and person1.moving == False:
                    #print('case2b in refresh, no current ROI, adjust box and meanshift')
                    
                    cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6) 
                    previousFX = person1.lastROICurrent[0]
                    previousFY = person1.lastROICurrent[1]
                    previousFW = person1.lastROICurrent[2]
                    previousFH = person1.lastROICurrent[3]
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                   
                    if person1.fX < previousFX: # code to keep box from wondering
                        person1.fX = previousFX
                        #print('fixed box location 9 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fX+person1.fW > previousFX+previousFW :
                        diff = (person1.fX+person1.fW) - (previousFX+previousFW)
                        if person1.fX > previousFX+diff:
                            person1.fX = person1.fX - diff
                            #print('fixed box location 10 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
                    if person1.fY < previousFY: # code to keep box from wondering
                        person1.fY = previousFY
                        #print('fixed box location 11 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    elif person1.fY+person1.fH > previousFY+previousFH:
                        diff = (person1.fY+person1.fH) - (previousFY+previousFH)
                        if person1.fY > previousFY+diff:
                            person1.fY = person1.fY - diff
                            #print('fixed box location 12 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
                
                    
                elif person1.edgeCounter > 0:
                    previousFX = person1.lastROICurrent[0]
                    previousFY = person1.lastROICurrent[1]
                    previousFW = person1.lastROICurrent[2]
                    previousFH = person1.lastROICurrent[3]
                    Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                   
                    if person1.fX < previousFX or person1.fX+person1.fW > previousFX+previousFW: # code to keep box from wondering
                        person1.fX = previousFX
                        #print('fixed box location 13 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                   
                    if person1.fY < previousFY or person1.fY+person1.fH > previousFY+previousFH: # code to keep box from wondering
                        person1.fY = previousFY
                        #print('fixed box location 14 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    
            elif len(person1.roiCurrent) == 0 and flag == 1:# : # do for every person with no BS roi and shares previous roicurrent
                person1.roiCounter += 1
                #print('case2a in refresh, no current ROI, adjust box and meanshift')
                cv2.rectangle(imgCopy, (person1.fX, person1.fY), (person1.fX+person1.fW, person1.fY+person1.fH), (0,128,255), 6) 
                previousFX = person1.lastROICurrent[0]
                previousFY = person1.lastROICurrent[1]
                previousFW = person1.lastROICurrent[2]
                previousFH = person1.lastROICurrent[3]
                Tools.peopleMeanshift(img,imgCopy,person1,WIDTH,HEIGHT,term_crit,ROI_RESIZE_DIM)
                                
                if person1.fX < previousFX: # code to keep box from wondering
                    person1.fX = previousFX
                    #print('fixed box location 15 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                elif person1.fX+person1.fW > previousFX+previousFW :
                    person1.fX = previousFX
                    #print('fixed box location 16 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                
                if person1.fY < previousFY: # code to keep box from wondering
                    person1.fY = previousFY
                    #print('fixed box location 17 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                elif person1.fY+person1.fH > previousFY+previousFH:
                    person1.fY = previousFY
                    #print('fixed box location 18 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')         
                    
            else: # person.roicurrent != [] and not shared
                #print('case3 in refresh, person has current ROI, personbox = person.roi',person1.ID)
                person1.sharedROI = False
                person1.roiCounter = 0
                box = [person1.roiCurrent[0], person1.roiCurrent[1], person1.roiCurrent[2], person1.roiCurrent[3]]
                person1.lastGoodROI = person1.roiCurrent
                person1.fX=box[0]
                person1.fY=box[1]
                person1.fW=box[2]
                person1.fH=box[3]
                
            person1.location.append([frameNumber,(person1.fX+(person1.fX+person1.fW))/2,(person1.fY+(person1.fY+person1.fH))/2])            
            if frameNumber % 1 == 0:     
                if len(person1.kalmanLocation) > 0:
                    world = Tools.pixelToWorld((person1.kalmanLocation[-1][0],person1.kalmanLocation[-1][1]),homography) # for movement classification
                    worldArray = np.array([world[0],world[1]],ndmin = 2)
                    worldArray.shape = (1,2)
                    person1.locationArray = np.append(person1.locationArray,worldArray, axis = 0) # for movement classification
                    person1.worldLocation = worldArray

            if person1.fX <= 25 or person1.fX+person1.fW >= WIDTH-1 or person1.fY+ person1.fH <= 50 or person1.fY+person1.fH >= HEIGHT-1 : #check if person1 is on edge of scene
                person1.nearEdge = True
                person1.edgeCounter +=1
            else:
                person1.nearEdge = False
            
            personList.remove(person1)
            
        
    def insertPerson(self,person,personList): # perhaps a better way to do this or it is unnessessary
        #print(len(personList),'person list length before')
        personList.append(person)
        personList.sort(key=lambda x: x.ID, reverse=False)

        
    def removePerson(self,personID,personList):
        i = 0
        if len(personList) > 0: #remove correct person from person list
            while i < len(personList):
                currentID = personList[-(i+1)].ID
                if personID == currentID:
                    personList.remove(personList[-(i+1)])
                    break
                i += 1
    
    def getPerson(self,personID,personList):
        i = 0
        if len(personList) > 0: #remove correct person from person list
            while i < len(personList):
                currentID = personList[-(i+1)].ID
                if personID == currentID:
                    return personList[-(i+1)]
                i += 1
            return []
        else:
            return []

# This class stores all information about a single person/object in the frame.     
class Person():
    
    def __init__(self,ID,fX,fY,fW,fH,visible,roiHist,height):
        self.ID=ID
        self.fX=fX
        self.fY=fY
        self.fW=fW
        self.fH=fH
        self.V=visible
        self.location=[] 
        self.kalmanLocation = []
        self.height = -1
        self.heightList = []
        self.width = -1
        self.widthList = []
        self.direction = []
        self.roiHist = roiHist
        self.roiHistList = [roiHist]
        self.kalmanX = Tools.KalmanFilter(fX+(fW/2),kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0) 
        self.kalmanY = Tools.KalmanFilter(fY+(fH/2),kalmanGain = 0,covariance = 1.0,measurmentNoiseModel = 1.5,covarianceGain = 1.10,lastSensorValue = 0) 
        self.roiCurrent = []
        self.lastROICurrent = []
        self.lastGoodROI = []
        self.locationArray = np.array([],ndmin = 2)
        self.locationArray.shape = (0,2)
        self.intersected = False                            #
        self.moving = True                                  #
        self.running = False                                #
        self.speed = -1                                  #
        self.clusterID = 0
        self.leftObject = []                                #            #list of tuples with frame num and location of object                          
        self.nearEdge = False                               #for detecting that the person is leaving the scene
        self.edgeCounter = 0                                #for detecting that the person is leaving the scene
        self.roiCounter = 0
        self.heading = -1
        self.worldLocation = []
        self.clusterGroup = None
        self.sharedROI = False


class ClusterGroup(): #class to hold the individual groups of occluded people. consider renaming these classes
    
    def __init__(self,label):
        self.index=0
        self.people=[]
        self.previousLen = 0
        self.label = label
        self.state = 1 #used to get rid of cluster that is empty
        
    
    def add(self, person):
        if len(person) <= 1:
            self.people.append(person)
            self.index= self.index + 1
        else:
            self.people.extend(person)
            self.index = self.index + len(person)
    
    def remove(self, person):
        if len(person) <= 1:
            self.people.remove(person)
            self.index= self.index - 1
        else:
            for person2 in self.people:
                if person2 in person:
                    self.people.remove(person)
                    self.index= self.index - 1    
                    
           
        
          
            
            
                
                
        
