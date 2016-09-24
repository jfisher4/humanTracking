import Tools

class CrowdBehaviorDetector():
    def __init__(self):
        self.previousState = None
        
        
    def updateState(self,peopleList,clusters,DC):
        
        results = []
        for clusterGroup in clusters:
            clusterID = clusterGroup.label
            northCounter = 0
            southCounter = 0
            eastCounter = 0
            westCounter = 0        
            movingCounter = 0
            runningCounter = 0
            oppositionCounter = 0
            speedConstancy = False
            headingConstancy = False
            clusterSpeedList = []
            headingList = []
            if len(clusterGroup.people) == 1:
                results.append(['group ',clusterID,'Singleton Group'])
                continue
            for person in clusterGroup.people:
                if person.speed != -1: #check speed
                    clusterSpeedList.append(person.speed)
                if person.heading != -1:
                    headingList.append(person.heading)
                    if person.heading > 315 or person.heading < 45: # person moving North
                        northCounter += 1
                        oppositionCounter += 1
                    elif person.heading > 135 and person.heading < 225: # person moving South
                        southCounter += 1
                        oppositionCounter += 1
                    elif person.heading >= 45 and person.heading <= 135: # person moving East
                        eastCounter += 1
                        oppositionCounter += 1
                    elif person.heading >= 225 and person.heading <= 315: # person moving West
                        westCounter += 1
                        oppositionCounter += 1
                if person.moving == True:
                    movingCounter += 1
                if person.running == True:
                    runningCounter += 1
            
            if len(clusterSpeedList)>1: #only consider if there are more than 1 person with a valid speed
                clusterSpeedList.sort()
                speedArray = Tools.np.array(clusterSpeedList)
                speedStdDev = Tools.np.std(speedArray)
                #speedMedian = Tools.np.median(speedArray)
                speedAverage = Tools.np.mean(speedArray)
                if speedStdDev/speedAverage <  .3: #need to fix this
                    #constancy in speed
                    speedConstancy = True
            
            if len(headingList)>1:    
                headingList.sort()
                headingArray = Tools.np.array(headingList)
                headingStdDev = Tools.np.std(headingArray)
                #headingMedian = Tools.np.median(headingArray)
                headingAverage = Tools.np.mean(headingArray)   
                if headingStdDev/headingAverage < .3:
                    headingConstancy = True
            
            
                
            
            
            
            
            
            
#            if speedConstancy == True and headingConstancy == True and movingCounter >= .5 * len(clusterGroup.people) :    
#                results.append(['group ',clusterID,'Constancy'])  
            
            #elif headingConstancy == False and movingCounter > .65 * len(clusterGroup.people) and runningCounter < .5 * len(clusterGroup.people):
                #results.append(['group ',clusterID,'Dispersion']) 
            
            #elif headingConstancy == False and movingCounter > .5 * len(clusterGroup.people) and runningCounter > .5 * len(clusterGroup.people):
                #results.append(['group ',clusterID,'Dispersion Under Duress'])     
                
            if (movingCounter < .5*len(clusterGroup.people)) and clusterGroup.previousLen == len(clusterGroup.people):#< .25*lengthOfCluster): #if the majority of members are not moving and part of a cluster then a meeting is taking place
                results.append(['group ',clusterID,'fixed meeting'])
            #check for varying meeting
            elif (movingCounter < .5*len(clusterGroup.people)) and clusterGroup.previousLen != len(clusterGroup.people): #if one of the members are moving and part of a cluster then a varying  meeting is taking place 
                results.append(['group ',clusterID,'varying meeting'])
        
            #if headingConstancy == False and movingCounter > .5 * len(clusterGroup.people) and oppositionCounter == 2:
                #results.append(['group ',clusterID,'Opposition']) 
            #elif Tools.np.DC == 1 and DC != 0.0:
            elif Tools.np.isclose([DC],[1.0],rtol=1e-03, atol=1e-04, equal_nan=False):
                results.append(['group ',clusterID,'Constancy']) 
            elif DC < 1 and DC != 0.0:
                results.append(['group ',clusterID,'Converging'])
            elif DC > 1 and DC != 0.0:
                results.append(['group ',clusterID,'Dispersion'])
            #elif Tools.np.DC == 1 and DC != 0.0:
            
                
        

    
        return results
    
    
    
    
    
    
    
    
    


    def findIntersection(self,people,lostPeople,imgDisplay,homography):
        
        #Find intersections with or without space and time
        
        ind = 0
        count = 0
        timeAndSpace = False
        tmpList = []
        coincidenceGroup = []
        returnReport = []
#        boundingRectangle = Tools.np.array([],ndmin = 2).astype(Tools.np.int0)
#        boundingRectangle.shape = (0,2)          #this is used later in determining meet, dispersion...
        for person in people:               #get all people and put in tmp list
            tmpList.append(person)
        for person in lostPeople:           #get all lost people and put in tmp list
            tmpList.append(person)
        
    
        for person in tmpList: #person may be a person or a group 
            if len(person.locationArray) > 0:
                imgPoints = Tools.worldToPixel((person.locationArray[-1][0],person.locationArray[-1][1]),homography,imgDisplay)
                imgArray = Tools.np.array([imgPoints[0],imgPoints[1]],ndmin = 2)
                imgArray.shape = (1,2)
#                boundingRectangle = Tools.np.append(boundingRectangle, imgArray, axis = 0) # used later in meet, dispersion...
#                print(boundingRectangle)
                
                Tools.locationTracker(person,imgDisplay) #draw zig zag line
                #Tools.findAverageHeading(imgDisplay,self.frameNumber,person,self.homography)
                #Tools.kfAdjustment(person,person.stepsSinceSensor)
                #if self.frameNumber % 10 == 0:
                    #Tools.kfAdjustment(person,person.stepsSinceSensor)
                if len(person.locationArray) > 1:
                    ind2 = 0
                    for person2 in tmpList:
                        
                        if ind != ind2:
                            if len(person2.locationArray) > 1:
                                intersection = Tools.findIntersections(person.locationArray,person2.locationArray) #find the intersection of the two people
                                #print('intersection',intersection)
                                
                                if len(intersection[0]) != 0 and len(intersection[1]) != 0:
                                    if len(intersection[0])< len(intersection[1]):
                                        intersectionLength = intersection[0]
                                    else:
                                        intersectionLength = intersection[1]
                                    for i in range(len(intersectionLength)):
                                        pixel = Tools.worldToPixel((intersection[0][i],intersection[1][i]),homography,imgDisplay)
                                        tempText = 'Intersection found.'
                                        Tools.cv2.putText(imgDisplay,tempText,(pixel),0, 1, (255,0,0), 2,4, False)
                                        count += 1
                                        person.intersected = True
                                        person2.intersected = True
                                        if ["intersection",str(person.ID),str(person2.ID)] not in returnReport:
                                            returnReport.append(["intersection",str(person.ID),str(person2.ID)])
                                        
                                    
                                        
                        ind2 += 1
            ind += 1
        if count == 0 and timeAndSpace == False:
            tempText = 'No intersection found.'
            Tools.cv2.putText(imgDisplay,tempText,(800,400),0, 1, (0,0,255), 2,4, False)
            
        
        if count > 0 and timeAndSpace == True and coincidenceGroup != []:
            tempText = 'Intersection In Time and Space.'
            position = (coincidenceGroup.location[-1][1],coincidenceGroup.location[-1][2])
            Tools.cv2.putText(imgDisplay,tempText,position,0, 1, (0,0,255), 2,4, False)  
            
            returnReport.append(["Intersection In Time and Space",str(coincidenceGroup.ID)])                    
                    
        #determine meet
                    
        #rect = Tools.cv2.boundingRect(boundingRectangle,)   
#        if len(boundingRectangle)>1:
#            print(boundingRectangle)#.astype(Tools.np.int0))                                # only try to find the rect if there is more than one person 
#            rect = Tools.cv2.minAreaRect(boundingRectangle)
#            #iCoordMin = Tools.worldToPixel((rect[0],rect[1]))
#            #iCoordMax = Tools.worldToPixel((rect[0]+rect[2],rect[1]+rect[3]))
#            #rect = [iCoordMin[0],iCoordMin[1],iCoordMax[0]-iCoordMin[0],iCoordMax[1]-iCoordMin[1]]
#            box = Tools.cv2.boxPoints(rect)
#            box = Tools.np.int0(box)
#            Tools.cv2.drawContours(imgDisplay,[box],0,(204,153,255),2)            
            #Tools.cv2.rectangle(imgDisplay, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (128,255,90), 2)          
                        
        return returnReport          
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
    