import cv2
import os
import numpy as np

# Global variables.
gx=0
gy=0
visible=False
init=False

# Helper function that listens for mouse actions for the VideoCropperHelper method.    
def draw_roi(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global gx,gy,visible,init
        gx=x
        gy=y
        visible=True
        init=True

# Cycles through video files in working directory to extract positive and negative sample images from.        
def videoCropper(inputFile):
    #for files in os.listdir(os.getcwd()):
        #print files
        #if files.endswith(".mp4") or files.endswith(".wmv") or files.endswith(".avi") or files.endswith(".MOV"):
            #print(os.getcwd()+"/"+files)
    videoCropperHelper(inputFile)
        #else:
            #print("invalid file")
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
def videoCropperHelper(inputfile):
    cv2.namedWindow('frame')
    cap = cv2.VideoCapture(inputfile)
    print(cap[0])
    global gx,gy,visible,init
    gx=0
    gy=0
    visible=False
    init=False
    frameno=0
    jmpframe=int(raw_input("Enter frame number to start(0 to start at beginning):"))
    while (frameno!=jmpframe):
        ret,frame=cap.read()
        frameno=frameno+1
    while(cap.isOpened()):
        width=64
        height=128
        ret, frame = cap.read()
        #print(frame)
        tmpframe=np.array(frame)
        tmpframe2=np.array(frame)
        cv2.setMouseCallback('frame',draw_roi)
        print("Frame number: "+ str(frameno))

        while(True):
            cv2.imshow('frame',tmpframe)

            if (visible):
                if(init):
                    tmpframe=np.array(frame)
                    cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                    init=False
            k = cv2.waitKey(10) & 0xFF
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a positive image.
            if k== ord(' ') and visible:
                print("Positive")
                roi=frame[gy:gy+height,gx:gx+width]
                resized_roi = cv2.resize(roi, (64,128))
                saveName=inputFile[0:-4]+"_"+str(frameno)+ "_" + str(gx)+ "_" + str(gy) + "_P.jpeg"
                cv2.imwrite(saveName,resized_roi)
                break
# Save Image for Homography 
            if k== ord('s') and visible:
                print("Homography Image")
                print(frame)
                #roi=frame[gy:gy+height,gx:gx+width]
                #resized_roi = cv2.resize(roi, (64,128))
                saveName=inputFile[0:-4]+"_"+str(frameno)+".jpeg"
                cv2.imwrite(saveName, tmpframe2)
                break
# Save roi defined by mouse as a 64x128 JPEG image; Image name is annotated as a negative image.
            if k== ord('n') and visible:
                print("Negative")
                roi=frame[gy:gy+height,gx:gx+width]
                resized_roi = cv2.resize(roi, (64,128))
                saveName=inputFile[0:-4]+"_"+str(frameno)+ "_" + str(gx)+ "_" + str(gy) + "_N.jpeg"
                cv2.imwrite(saveName,resized_roi)
                break
# Decreases size of roi.
            if k== ord(','):
                if width>=16:
                    width=width-4
                    height=height-8
                    tmpframe=np.array(frame)
                    cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                    print("Width:"+str(width)+ " Height:"+str(height))
# Increases size of roi.
            if k== ord('.'):
                if width<=128:
                    width=width+4
                    height=height+8
                    tmpframe=np.array(frame)
                    cv2.rectangle(tmpframe,(gx,gy),(gx+width,gy+height),(255,0,0),1)
                    print("Width:"+str(width)+ " Height:"+str(height))
# Skips to next frame.
            if k== ord('v'):
                break
# q for quit            
            if k== ord('q'):
                cap.release()
                break

        visible=False
        frameno=frameno+1
    cap.release()
    cv2.destroyAllWindows()

def displayOptions():
    print("=============================================================================================")
    print("----Options----")
    print("1. Create positive and negative samples from videos in given directory")
    print("2. Exit")

# Start of script.    

inputFile= "/home/robotics_group/seniorProject/01072016A5_J1.mp4"        #raw_input("Enter the name of the directory where the files are located:\n")

end=False
displayOptions()
while(not end):
    inputCommand=raw_input("Ready for command: ")

    if(inputCommand=="1"):
        videoCropper(inputFile)
        displayOptions()

    elif(inputCommand=="2"):
        end=True
    else:
        print("Not a valid command. Try Again...")
        displayOptions()


print("Ending program...")

