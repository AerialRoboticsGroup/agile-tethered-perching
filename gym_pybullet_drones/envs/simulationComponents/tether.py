import numpy as np
import pybullet as p
from typing import List, Any
import random
import math

'''reference:https://github.com/TommyWoodley/TommyWoodleyMEngProject'''

class Tether:
    RADIUS = 0.006
    # MASS = 0.05  # for 1 meter
    MASS= 0.00001
    
    def get_radius(self):
        return self.RADIUS

    def __init__(self, drone_id: Any, length: float, drone_position:np.ndarray, physics_client: int, num_segments: int = 20) -> None:
        assert isinstance(length, float), "length must be an instance of float"
        assert isinstance(drone_position, np.ndarray), "top_position must be an instance of np.ndarray"
        assert isinstance(physics_client, int), "physics_client must be an instance of int"
        assert isinstance(num_segments, int), "num_segments must be an instance of int"

        self.physics_client = physics_client
        self.length = length
        self.num_segments = num_segments
        self.segment_length = length / num_segments
        self.segment_mass = self.MASS / self.num_segments  # Distribute the mass across the segments
        # print(self.segment_mass)
        self.segments = []
        
        # bottom end
        self._parent_frame_pos = np.array([0, 0, -0.5 * self.segment_length], dtype=np.float32)
        # top end
        self._child_frame_pos = np.array([0, 0, 0.5 * self.segment_length], dtype=np.float32)

        self.drone_position= np.array(drone_position)  # Initialize drone position
        # print(self.drone_position)
        self.offset = 0.02
        self.drone_bottom_offset = np.array([0, 0, -self.offset])
        
        self.top_position = self.drone_position + self.drone_bottom_offset
            
        self.create_tether()

        self.one_third_index =  (len(self.segments) // 3 )*2 + 1 #6*2+1=13
        # one_third_index = len(self.segments) // 4 * 3
        self.one_third_indexs = [self.one_third_index, self.one_third_index+1]

        self.weight_prev_angle = None
        
        self.weight_cumulative_angle_change = 0.0
        self.weight_wraps = 0.0
 
        self.time = 0
    

    def create_tether(self) -> None:
        current_position = self.top_position[:]
        # print(current_position)
        self.segment_base_orientation = None
        # Create each segment
        for i in range(self.num_segments):
           
            if current_position[2] - self.segment_length <= 0:
                
                if i == 0:
                    segment_top_position = [
                    current_position[0]-self.offset,
                    current_position[1],
                    0
                    ]
                    
                    
                    self.segment_base_orientation = [0,0,1]
                else:
                    segment_top_position = [
                        current_position[0] - self.segment_length,
                        current_position[1],
                        0
                    ]
                     #along x
                    self.segment_base_orientation = [0, math.pi/2, 0]
                    # self.segment_base_orientation = [0, 0, 1]
                segment_base_position = [
                    segment_top_position[0] - 0.5 * self.segment_length,
                    segment_top_position[1],
                    0
                    ]
                   
            else:
                if i == 0:
                    segment_top_position = current_position
                else:
                    # Vertical descent when above ground
                    segment_top_position = [
                        current_position[0],
                        current_position[1],
                        current_position[2] - self.segment_length
                    ]
            
                segment_base_position = [
                        segment_top_position[0],
                        segment_top_position[1],
                        segment_top_position[2]-0.5 * self.segment_length
                    ]
                #along z
                self.segment_base_orientation = [0,0,1]
            
            # Debugging print statements
            # print(f"{i} top position: {segment_top_position}")
            # print(f"{i} base position: {segment_base_position}")

            # Collision and visual shapes
            collisionShapeId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.RADIUS, height=self.segment_length)
            visualShapeId = p.createVisualShape(p.GEOM_CYLINDER, radius=self.RADIUS,
                                                length=self.segment_length, rgbaColor=[1, 0, 0, 1])
            
            # last_one_third_point = self.get_last_one_third_point()
    
        
            sphere_visual = p.createVisualShape(
               p.GEOM_CYLINDER, radius=self.RADIUS,
                                                length=self.segment_length, rgbaColor=[0, 1, 0, 1]
            )

      
            # Create the segment
            if i == 13:
                
                segment_id = p.createMultiBody(baseMass=self.segment_mass,
                                        baseCollisionShapeIndex=collisionShapeId,
                                        baseVisualShapeIndex=sphere_visual,
                                        basePosition=segment_base_position,
                                        baseOrientation=p.getQuaternionFromEuler(self.segment_base_orientation))

            else:    
                segment_id = p.createMultiBody(baseMass=self.segment_mass,
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=segment_base_position,
                                            baseOrientation=p.getQuaternionFromEuler(self.segment_base_orientation))

            self.segments.append(segment_id)
            # p.changeDynamics(segment_id, -1, lateralFriction=1.2, linearDamping=0.0, angularDamping=0.0)
            p.changeDynamics(segment_id, -1, 
                 lateralFriction=1.2,       # More friction between segments
                 rollingFriction=0.02,       # Prevent rolling
                 spinningFriction=0.02,      # Prevent free spinning
                 linearDamping=0.0,        # Reduce linear velocity over time
                 angularDamping=0.0)       # Reduce rotational velocity over time

            # Connect this segment to the previous one (if not the first)
            if i > 0:
                self.create_rotational_joint(
                    parent_body_id=self.segments[i - 1],
                    child_body_id=segment_id,
                    parent_frame_pos=self._parent_frame_pos,
                    child_frame_pos=self._child_frame_pos
                )
            
            # Update the current position for the next segment
            current_position = segment_top_position
            
    
    
    def attach_to_drone(self, drone_id: Any) -> None:
        # Convert drone_id to int if it's not already
        drone_id = int(drone_id)

        # Use the create_fixed_joint function to attach the top segment to the drone
        self.create_fixed_joint(
            parent_body_id=drone_id,
            child_body_id=self.segments[0],
            parent_frame_pos= self.drone_bottom_offset,
            child_frame_pos=[0, 0, self.segment_length / 2]
        )
        
    def attach_weight(self, weight: Any) -> None:
        # Attach the weight to the bottom segment
        tether_attachment_point = self._parent_frame_pos
        weight_attachment_point = weight.get_body_centre_top()
   
        self.create_fixed_joint(parent_body_id=self.segments[-1],  # Bottom segment
                                child_body_id=weight.weight_id,
                                parent_frame_pos=tether_attachment_point,
                                child_frame_pos=weight_attachment_point)

    def create_rotational_joint(self, parent_body_id: int, child_body_id: int, parent_frame_pos: np.ndarray,
                                child_frame_pos: np.ndarray) -> None:
        assert isinstance(parent_body_id, int), "parent_body_id must be an instance of int"
        assert isinstance(child_body_id, int), "child_body_id must be an instance of int"
        assert isinstance(parent_frame_pos, np.ndarray), "parent_frame_pos must be an instance of np.ndarray"
        assert isinstance(child_frame_pos, np.ndarray), "child_frame_pos must be an instance of np.ndarray"

        # Use a point-to-point joint to connect the segments
        p.createConstraint(parentBodyUniqueId=parent_body_id,
                           parentLinkIndex=-1,
                           childBodyUniqueId=child_body_id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_POINT2POINT,
                           jointAxis=[0,0,1],
                           parentFramePosition=parent_frame_pos.tolist(),
                           childFramePosition=child_frame_pos.tolist())

    def create_fixed_joint(self, parent_body_id: int, child_body_id: int, parent_frame_pos: Any,
                           child_frame_pos: Any) -> None:
        assert isinstance(parent_body_id, int), "parent_body_id must be an instance of int"
        assert isinstance(child_body_id, int), "child_body_id must be an instance of int"
        assert isinstance(parent_frame_pos, (List, np.ndarray)), "wrong type"
        assert isinstance(child_frame_pos, (List, np.ndarray)), "wrong type"

        p.createConstraint(parentBodyUniqueId=parent_body_id,
                           parentLinkIndex=-1,
                           childBodyUniqueId=child_body_id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=[0, 0, 1],
                           parentFramePosition=parent_frame_pos.tolist() if isinstance(parent_frame_pos, np.ndarray) else parent_frame_pos,
                           childFramePosition=child_frame_pos.tolist() if isinstance(child_frame_pos, np.ndarray) else child_frame_pos,
                           parentFrameOrientation=[0, 0, 0, 1],
                           childFrameOrientation=[0, 0, 0, 1])

    def get_segments(self):
        return self.segments
    
    def get_avg_point(self, index_list):
        
        positions = [p.getBasePositionAndOrientation(obj_id)[0] for obj_id in
                     (self.segments[i] for i in index_list)]

        # Calculate the midpoint
        midpoint = [(pos1 + pos2) / 2 for pos1, pos2 in zip(positions[0], positions[1])]
        return midpoint
  
    def get_last_one_third_point(self):
        last_one_third_point = self.get_avg_point(self.one_third_indexs)
        # print(self.one_third_indexs, last_one_third_point)
        return last_one_third_point

    def get_world_centre_bottom(self) -> np.ndarray:
        # return np.array([-self.length, 0.0, 0.0])
        bottom_position, _ = p.getBasePositionAndOrientation(self.segments[-1])
        bottom_position = list(bottom_position)  # Convert to list to allow modifications
        # print(f"Initial bottom_position: {bottom_position}")
        
        if bottom_position[2] == 0.0:
            bottom_position = np.array([-self.segment_length/2.0, 0.0, 0.0], dtype=np.float32) + np.array(bottom_position, dtype=np.float32)
        elif bottom_position[2] > 0.0:
            bottom_position[2] = bottom_position[2] - self.segment_length * 0.5
        if bottom_position[2] < 0.0:
            bottom_position[2]=0.0
            # raise ValueError("The bottom position of the tether should not be negative in the z-axis.")
        
        # print(f"Adjusted bottom_position: {bottom_position}")
        # print(self.drone_position)
        return np.array(bottom_position)

            
        