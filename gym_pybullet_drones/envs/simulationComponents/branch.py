import pybullet as p
import numpy as np
from typing import List

class Branch:

    def __init__(self) -> None:
        self.branch_id = None
        self.branch_pos = None  # Initialize branch position to None

    def add_tree_branch(self, position: list, length: float = 2.0, radius: float = 0.01,
                        orientation: list = [np.pi / 2, 0.1, 0]) -> None:
        '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
        assert isinstance(position, list), "position must be an instance of List"
        assert isinstance(length, float), "length must be an instance of float"
        assert isinstance(radius, float), f"radius must be an instance of float, found:{type(radius)}"
        assert isinstance(orientation, list), "orientation must be an instance of List"
        self.radius = radius
        orientation_quat = p.getQuaternionFromEuler(orientation)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=radius,
                                              length=length, rgbaColor=[0.6, 0.32, 0.17, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=radius, height=length)
        self.branch_id = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=collision_shape_id,
                                           baseVisualShapeIndex=visual_shape_id,
                                           basePosition=position,
                                           baseOrientation=orientation_quat)
        self.branch_pos = position
        p.changeDynamics(self.branch_id, -1, lateralFriction=1.0)

    def get_tree_branch_midpoint(self):
        return self.branch_pos

    def get_body_id(self):
        return self.branch_id
        
    def get_branch_circumference(self):
        
       
        return 2 * np.pi * self.radius

