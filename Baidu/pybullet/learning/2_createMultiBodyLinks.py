import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)

sphereRadius = 0.2
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

mass = 1
visualShapeId = -1

link_Masses = [1]

linkVisualShapeIndices = [-1]
linkPositions = [[0, 0, 0.11]]
linkOrientations = [[0, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[0, 0, 1]]


for j in range(2):
  for k in range(2):
    basePosition = [
        0, 1 + j * 2 * sphereRadius + k * sphereRadius, 1 + k * 2 * sphereRadius + 1
    ]
    baseOrientation = [0, 0, 0, 1]
    # you can create a multi body using createMultiBody
    sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
                                  baseOrientation)
    # You can change the properties such as mass, friction and restitution coefficients using changeDynamics.
    p.changeDynamics(sphereUid,
                      -1,
                      spinningFriction=0,
                      rollingFriction=0,
                      linearDamping=0)

    for joint in range(p.getNumJoints(sphereUid)):
      p.setJointMotorControl2(sphereUid, joint, p.VELOCITY_CONTROL, targetVelocity=100, force=10)

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

p.getNumJoints(sphereUid)
for i in range(p.getNumJoints(sphereUid)):
  p.getJointInfo(sphereUid, i)

while (1):
  keys = p.getKeyboardEvents()
  print(keys)

  time.sleep(0.01)
