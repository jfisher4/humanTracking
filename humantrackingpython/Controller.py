from Simulation import Simulator
import cv2


class PeopleController:
    """
    jdsjfdsjkdsjfsdasjkdsahkjhskjdashkdhskahsdjsadna
    @brief aaaaaaaaaaaaaaaaaaaaaa
    """
    def __init__(self):
        self.A = None
        self.B = None
        
    def update(self,peopleA,peopleB):
        pass   

def testSingleView(directory,videoname):
    simulationA = Simulator(directory,videoname)
    active = 1 # 1 for active , 0 for inactive, 2 for paused
#    active = True
#    paused = False
    while(1): 
        if active == 1:
            #peopleA, active, paused = simulationA.retrieve(paused)
            peopleA, active = simulationA.retrieve()
        elif active == 0:
            break
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            print("Continuing...")
            #paused = False
            active = 1
        elif k == ord('q'):
            print("Exiting Program...")
            
            cv2.destroyAllWindows()
            break

def testMultiView(directory,videoname1,videoname2):
    simulationA = Simulator(directory,videoname1)
    simulationB = Simulator(directory,videoname2)
    controller = PeopleController() 
    active = True
    while(active):
        peopleA, active = simulationA.retrieve()
        peopleB, active = simulationB.retrieve()

if __name__ =="__main__":
    testSingleView("C:\\Users\\James\\Documents\\Millennium\\Pinos\\multiview\\","01072016A5_J1.mp4") #05282015A5_A21.mp4, 12102015A5_E1.mp4, 12102015A5_B1.mp4
    #testMultiView("C:\\Millennium\\Videos\\05282015\\","05282015A5_A1.mp4","05282015B5_A1.mp4")
