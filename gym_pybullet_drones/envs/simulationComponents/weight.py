import pybullet as p
from typing import List
import numpy as np

class Weight:
    #0.000001
    MASS: float = 0.000001
    RADIUS: float = 0.03

    _body_centre_top = np.array([0, 0, RADIUS], dtype=np.float32)

    def __init__(self, top_position: np.ndarray, mass = 0.000001) -> None:
        assert isinstance(top_position, np.ndarray), "top_position must be an instance of np.ndarray"

        top_x, top_y, top_z = top_position
        if top_z == 0:
            self.base_position = [top_x - self.RADIUS, top_y, top_z + self.RADIUS]
        elif top_z > 0:
            self.base_position = [top_x, top_y, top_z - self.RADIUS]
        else:
            raise ValueError("The payload connection point should not be negative in the z-axis.")
        self.mass_weight = mass
        self.create_weight(mass)
        self.cross_area = 3 * self.RADIUS * self.RADIUS

    def create_weight(self,mass=0.00001) -> None:
        
        '''https://github.com/TommyWoodley/TommyWoodleyMEngProject'''
        collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=self.RADIUS)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=self.RADIUS, rgbaColor=[0, 0, 1, 1.0])

        self.weight_id = p.createMultiBody(baseMass=mass,
                                           baseCollisionShapeIndex=collisionShapeId,
                                           baseVisualShapeIndex=visualShapeId,
                                           basePosition=self.base_position,
                                           baseOrientation=[0, 0, 0, 1])

    def get_position(self) -> List[float]:
        position, _ = p.getBasePositionAndOrientation(self.weight_id)
        return position

    def get_body_centre_top(self) -> np.ndarray:
        return self._body_centre_top
    
    def get_weight_id(self):
        return self.weight_id