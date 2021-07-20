import pybullet as p
import time
import pybullet_data
import os
import random


for sum_100 in range(10):
  p.connect(p.GUI)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.createCollisionShape(p.GEOM_PLANE)
  p.createMultiBody(0, 0)

  sphereRadius = 0.1
  colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

  mass = 1
  visualShapeId = -1

  sum_sphereUid = []
  for j in range(2):
    for k in range(2):
      basePosition = [
          0,  j * 3 * sphereRadius + random.uniform(0,0.2), 0.5 + k * 3 * sphereRadius + random.uniform(0,0.2)
      ]
      baseOrientation = [0, 0, 0, 1]
      # you can create a multi body using createMultiBody
      sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
                                    baseOrientation)
      sum_sphereUid.append(sphereUid)
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

  def get_pos(x):
    _output = list(p.getBasePositionAndOrientation(x)[0])
    output = []
    output.append(round(_output[1],4))
    output.append(round(_output[2],4))
    return output

  list_dir = os.listdir("dataset\\")
  current_path = "dataset\\" # 新创建的txt文件的存放路径
  full_path = current_path + str(sum_100) + '.txt' # 也可以创建一个.doc的word文档
  file = open(full_path, 'a')

  for _ in range(100):
    cubePos = list(map(get_pos,sum_sphereUid))
    cubePos = str(cubePos)
    file.write(cubePos) #msg也就是下面的Hello world!
    file.write("\n")
    time.sleep(0.01)

  file.close()
  p.disconnect()
