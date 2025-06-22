## save trajectory in txt file
## x = 0; fly in y, z max 6m #
## z 4 m max delta  #
import pandas as pd


import numpy as np
from numpy import arctan, arcsin, arccos, tan, cos, sin, pi, sqrt
import matplotlib.pyplot as plt

from plotTargetedTijectory import plotFlyingTrajectory


def stepBigenough(lastX,lastZ,x,z,v,dt):
    distance = ((lastX-x)**2+(lastZ-z)**2)**0.5
    return distance >=(v*dt)

def createAndSaveTrajectory(angelDrone, droneVelocety, diameterTarget):
    ##Target Branch
    positionTarget = np.array([0,2.7])
    lengthFinalStraight = 0
    #drone
    velocety = droneVelocety
    LengthRope = 1
    distanceToTarget = 4*LengthRope
    print("distance is: ", distanceToTarget)
    n = 1
    massPendelum = 1
    ## global
    dt = 1/20

    ## fily to first point
    cycleX, cycleZ, finalPosDrone = plotFlyingTrajectory(distanceToTarget, angelDrone, diameterTarget, positionTarget, lengthFinalStraight)

    trajectoryX = np.array([[cycleX[-1]]])
    trajectoryZ = np.array([[cycleZ[-1]]])
    velocetyX = np.array([[0]])
    velocetyZ = np.array([[0]])
    lastX = cycleX[-1]
    lastZ = cycleZ[-1]
    length = cycleX.shape[0]
    iLast = length-2
    for i in range(length-2,0,-1):
        if(stepBigenough(lastX,lastZ,cycleX[i],cycleZ[i],velocety,dt)):
            relation = (cycleZ[i]-lastZ)**2/(cycleX[i]-lastX)**2
            tmpVX = sqrt(1/(1+relation))
            tmpVZ = sqrt(1-tmpVX**2)
            lastX = cycleX[i]
            lastZ = cycleZ[i]
            trajectoryX = np.append([[cycleX[i]]],trajectoryX,axis=0)
            trajectoryZ = np.append([[cycleZ[i]]],trajectoryZ,axis=0)
            velocetyX = np.append([[tmpVX]],velocetyX,axis=0)
            velocetyZ = np.append([[tmpVZ]],velocetyZ,axis=0)
    
    ## move and wait 2s at starting pos
    timeStepsInitilase = 1 # int(np.ceil(2/dt)) # wait 2s
    initPosX = np.ones((timeStepsInitilase,1))*trajectoryX[0]
    initPosZ = np.ones((timeStepsInitilase,1))*trajectoryZ[0]
    initVX = np.zeros((timeStepsInitilase,1)) ##one to much from above allways v to get to the next point 
    initVZ = np.zeros((timeStepsInitilase,1))

    trajectoryX = np.append(initPosX, trajectoryX[1:], axis=0)
    trajectoryZ = np.append(initPosZ, trajectoryZ[1:], axis=0)
    velocetyX = np.append(initVX, velocetyX[1:], axis=0)
    velocetyZ = np.append(initVZ, velocetyZ[1:], axis=0)

    # ## correct the relative position
    # trajectoryX = trajectoryX - cycleX[0]
    # if np.min(trajectoryZ)<0:
    #     trajectoryZ -= np.min(trajectoryZ)
    # trajectoryZ = trajectoryZ + 1 # savety hight above floor 

    # if(np.any(trajectoryZ>4) or np.any(abs(trajectoryX)>3)):
    #     print("Warning: position my be out of range for the Lab dimension! (6x4m)")

    ## going from 2D to 3D and adding angel
    addingZero = np.zeros((trajectoryX.shape[0],1))
    
    #print(addingDim.shape)
    trajectory = np.hstack((addingZero,trajectoryX,trajectoryZ,addingZero,velocetyX,velocetyZ,addingZero)) #switch her y back to z

    ### save txt file ###
    fileName = "./Trajectories/velocety" + str(int(velocety)) + "_droneAngel" + str(int(angelDrone*100)) + ".txt"
    note = ""
    np.savetxt(fileName,trajectory,header='xCoordinates yCoordinates zCoordinates vx vy vz yarnAngel',comments=note)

    fig = plt.figure(figsize=(8, 8/1.618))
    plt.rcParams.update({'font.size': 18})
    ax = plt.subplot(111)
    ax.plot(cycleX[0], cycleZ[0], 'rs', label='Start Pos')
    ax.plot(cycleX, cycleZ, 'k--', label='Trajectory')
    ax.plot(finalPosDrone[0], finalPosDrone[1], 'md', label='Final Pos') 
    ax.plot((cycleX[-1], finalPosDrone[0]), (cycleZ[-1],finalPosDrone[1]), 'k--') 
    #ax.plot(cycleX[-1], cycleZ[-1], 'b^') 
    plt.plot(positionTarget[0], positionTarget[1], 'go', label='Target Branch')
    #plt.plot(senterOfTrajectory[0], senterOfTrajectory[1], 'bx')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                      ncol=1, mode="expand", borderaxespad=0.)
    size = int(1.2*distanceToTarget)
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.set(xlabel='relative x position', ylabel='relative z position')
    
    ### save figures ###
    plt.draw()
    plt.savefig("./Trajectories/velocety" + str(int(velocety)) + "_droneAngel" + str(int(angelDrone*100))+".eps", dpi=fig.dpi, format='eps')

if __name__ == '__main__':
    ##Target Branch
    positionTarget = np.array([0,2.7])
    diameterTarget = 0.01
    ##drone 
    angelDrone = pi/8
    droneVelocety = 2
 
    createAndSaveTrajectory(angelDrone, droneVelocety, diameterTarget)
    
   
    # Load the TXT file (space-separated values)
    # df = pd.read_csv('/Users/kangleyuan/Downloads/tethered-perching-initial-main/IdealAngelAnalyticalSolution/Trajectories/velocety2_droneAngel39.txt', delim_whitespace=True, header=None)
    df = pd.read_csv('./Trajectories/velocety2_droneAngel39.txt', sep=r'\s+', comment='#')
    # Extract position (y, x, z) and velocity (vx, vy, vz)
    df_final = df.iloc[:, [1, 0, 2, 4, 3, 5]]  # y, x, z, vx, vy, vz

    # Rename columns accordingly (labels reflect swapped x-y)
    df_final.columns = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    # Save to CSV with commas and no index
    df_final.to_csv('../simple_baseline_traj.csv', index=False)
