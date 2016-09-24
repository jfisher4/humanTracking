# TODO: Write code in C++ 

import pickle
import os
import Tools
directory = raw_input("Enter the directory in which the pickle files are located: ")
os.chdir(directory)
videoname = raw_input("Enter the date and camera of video files: ")
homography = pickle.load( open( videoname+"_H.p", "rb" ) )
rotationMatrix = pickle.load(open( videoname+"_rotM.p","rb"))
tvec = pickle.load( open( videoname+"_tvec.p", "rb" ) )
cameraPosition = Tools.cameraPosition(rotationMatrix, tvec)
#print(homography)
#print(rotationMatrix)
#print(tvec)
print(cameraPosition)

f_cp = open(videoname+"_CP.txt","w")
f_cp.write("1\n")
f_cp.write("3\n")
for row in cameraPosition:
    for column in row:
        f_cp.write(str(column)[3:-2])
        f_cp.write("\n")
f_cp.close()
f_h = open(videoname+"_H.txt","w")
f_h.write("3\n")
f_h.write("3\n")
for row in homography:
    for column in row:
        f_h.write(str(column))
        f_h.write("\n")
f_h.close()
