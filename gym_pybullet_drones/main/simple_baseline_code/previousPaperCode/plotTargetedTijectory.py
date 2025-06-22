import matplotlib.pyplot as plt
import numpy as np
from numpy import arctan, arcsin, arccos, tan, cos, sin, pi, sqrt
from tabulate import tabulate
import math

def getYPosCenterTrajectory(XD,YD,XT,YT):
    """
    YC=(YT^2-YD^2-XD^2)/2/(YT-YD)
    """
    return (YT**2-YD**2-(XD-XT)**2)/2/(YT-YD)

def plotTrajectory(center,radius,startAngel = 0, finalAngel = 2*pi):
    """return points to plot a cycle"""
    t = np.linspace(startAngel, finalAngel, 100000)
    x = sin(t)*radius+center[0]
    y = cos(t)*radius+center[1]

    return x, y

def get_Angle(d,t,c): #enter three points return the angel
    #fix issue of angels greater pi
    if d[1]==t[1]:
        return pi/2 
    ang = arccos((d[1]-t[1])/sqrt((d[0]-t[0])**2+(d[1]-t[1])**2))

    if d[0]<t[0]:
        ang = 2*pi-ang
        return ang
    return ang

def plotFlyingTrajectory(distanceToTarget, angelDrone, diameterTarget, positionTarget = np.array([0,2.7]), lengthFinalStraight = 1):

    startPosDrone = np.array([1.97, 3.0])
    XD, YD = startPosDrone[0], startPosDrone[1]
    
    LengthRope = 1.0
    finalPosDrone = np.array([positionTarget[0],positionTarget[1]],dtype=np.float16) ## change this!

    if finalPosDrone[0]<XD:
        finalPosDrone[0] += lengthFinalStraight
    else:
        finalPosDrone[0] -= lengthFinalStraight
    
    finalPosDrone[1] += LengthRope/2  # drone stops dreckly above the target 


    senterOfTrajectory = np.array([finalPosDrone[0], getYPosCenterTrajectory(XD,YD,finalPosDrone[0],finalPosDrone[1])])

    radiusTrajectory = ((finalPosDrone[0]-senterOfTrajectory[0])**2+(finalPosDrone[1]-senterOfTrajectory[1])**2)**0.5
    
    #get angle for right starting point
    startAngel = (get_Angle((XD, YD),senterOfTrajectory,finalPosDrone))

    if YD > finalPosDrone[1]:
        endAngel = pi
        if abs(XD-positionTarget[0]) <= abs(positionTarget[0]-finalPosDrone[0]):            
            if startAngel < pi:
                startAngel = 2*pi+startAngel
            else:
                startAngel = startAngel-2*pi

    else:
        if XD < positionTarget[0]:
            endAngel = 2*pi
        else:
            endAngel = 0

    cycleX, cycleY = plotTrajectory(senterOfTrajectory,radiusTrajectory, startAngel,endAngel)

    return cycleX, cycleY, finalPosDrone

