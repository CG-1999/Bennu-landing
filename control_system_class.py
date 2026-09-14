import numpy as np
from Bennu_gravity_model import Bennu_main
from occupany_grid import build_and_save_occupancy_lowram
from a_star_map import astar_main
from path_pruning import main_PathPruned
from landing_map import LandingMap_main
from path_generation import PathGeneration_main
from Trajectory_gen import trj_main
from nominal_euler_angles_val import euler_nominal_main
from orientation import o_main
from nominal_ctrl_input import compute_nominal_control
from linearizer import compute_linearized_matrices
from controllability import compute_ctrl
from lqr import compute_dlqr
from control_sys import controller_main


class SpaceCraft():
    def __init__(self, mass, I, a, dfa, current_pos, current_orientation):
        self.mass = mass # Mass of the spacecraft
        self.I = I # Moment of inertia of the spacecraft
        self.a = a # Side length
        self.dist_from_asteroid = dfa
        self.current_pos = current_pos
        self.current_orientation = current_orientation

    def calculate_gravity_field(self):
        Bennu_main(self.dist_from_asteroid)

    def occupany_grid(self):
        build_and_save_occupancy_lowram(mesh_path="bennu.obj", step=30, padding=self.dist_from_asteroid, dilation_iter=3, 
                                        save_path="occupancy_grid.npz", max_workers=None, xy_batch_size=256)
        
    def landing_map(self):
        LandingMap_main()

    def astar_path(self):
        astar_main(mesh_path="bennu.obj", g_field_path="bennu_gravity_field30.npz", occ_path="bennu_occ.npz",
            start_point=self.current_pos, step=30)
    
    def path_pruning(self):
        main_PathPruned()

    def SmoothPath_generation(self):
        PathGeneration_main()

    def trajectory_generation(self):
        trj_main()

    def required_orientation(self):
        req_orientation = o_main()
        return req_orientation

    def nominal_euler_traj(self, req_orientation):
        euler_nominal_main(self.current_orientation, req_orientation)

    def nominal_input(self):
        compute_nominal_control(self.mass, self.I)

    def linearizer(self):
        compute_linearized_matrices(self.mass, self.I)

    def controllability(self):
        compute_ctrl()

    def lqr(self):
        # LQR weights
        Q_diag = [
            100, 100, 100,        # position
            500, 500, 500,        # attitude
            50, 50, 50,           # velocity
            200, 200, 200         # angular velocity
        ]

        Q = np.diag(Q_diag)
        R = np.eye(10) * 1.0 
        dt = 0.1
        compute_dlqr(Q,R,dt)

    def controller_(self):
        dt = 0.1
        extend_time_sec = 10
        controller_main(dt, extend_time_sec)



I = np.array([[6, 0, 0],
              [0, 6, 0],
              [0, 0, 6]])        
current_pos = np.array([-800.0, -900.0, 700.0])
current_orientation = np.array([0.0,0.0,0.0])
Krypto = SpaceCraft(50, I, 0.6, 1000, current_pos, current_orientation)
# Krypto.nominal_input()
# Krypto.linearizer()
# Krypto.controllability()
# Krypto.lqr()
# Krypto.controller_()
Krypto.astar_path()

