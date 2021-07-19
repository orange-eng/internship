import pybullet as p
from time import sleep
import pybullet_data

physicsClient = p.connect(p.GUI)    
p.setAdditionalSearchPath(pybullet_data.getDataPath())  #optionally
p.setGravity(0,0,-10)   # set the gravity
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("teddy_large.urdf",startPos,startOrientation)


# set the center of mass frame

for i in range(1000):
    p.stepSimulation()
    sleep(1./200.)

    cubePos,cubeOrn = p.getBasePositionAndOrientation(boxId)
    #print(cubePos,cubeOrn)
    print("CubePos=",cubePos)
    print("CubeOrn=",cubeOrn)
    # cubeOrn是一个四元数[x,y,z,w]，
    # 详细可看链接https://www.cnblogs.com/21207-iHome/p/6894128.html
p.disconnect()