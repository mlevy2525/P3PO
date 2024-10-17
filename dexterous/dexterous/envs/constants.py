from dataclasses import dataclass
import numpy as np


@dataclass
class Intrinsics:
    F_X: float
    F_Y: float
    C_X: float
    C_Y: float
    W: int
    H: int
    D: np.ndarray
    
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.F_X, 0, self.C_X], [0, self.F_Y, self.C_Y], [0, 0, 1]], dtype=np.float64)


CAM_TO_INTRINSICS = {
    "raybans": Intrinsics(
        F_X=2400, F_Y=2400, C_X=1512, C_Y=2016, W=3024, H=4032, D=np.zeros(5, dtype=np.float32),
    ),
    "realsense-141722071999": Intrinsics(
        F_X=900.57477747, F_Y=900.57477747, C_X=638.12883091, C_Y=366.73743594, W=1280, H=720, D=np.zeros(5, dtype=np.float32),
    ),
    "realsense-023422073116": Intrinsics(
        F_X=919.230285644531, F_Y=917.224609375, C_X=643.210754394531, C_Y=371.862487792969, W=1280, H=720, D=np.zeros(5, dtype=np.float32),
    ),
    # these are intrinsics estimated by eyeballing the estimation
    "realsense": Intrinsics(
        F_X=900, F_Y=900, C_X=640, C_Y=360, W=1280, H=720, D=np.zeros(5, dtype=np.float32),
    ),
}

FRANKA_HOME_CARTESIAN = np.array(
    # [ 0.7695382, 0.09684828, 0.40865096, 0.02469655, -0.71643, -0.6962282, 0.03721225]
    # [ 0.42841426, -0.17159191,  0.2939817 ,  0.25004667,  0.79409593, 0.51222575,  0.21098118]
    # [0.28145096, -0.19354133,  0.4024293 ,  0.41639453,  0.71167785, 0.3818502 ,  0.41751724]
    [ 0.4653203 , -0.3000841 ,  0.31952816,  0.19849579,  0.8202563 , 0.49294227,  0.21162905]
)
KINOVA_HOME_CARTESIAN = np.array(
    [0.09782694, -0.4715094, 0.37106162, 0.5673449, -0.65465486, 0.295827, 0.40253347]
)
ALLEGRO_HOME = np.array(
    [
        -0.0124863,
        -0.10063279,
        0.7970152,
        0.7542225,
        -0.01191735,
        -0.10746645,
        0.78338414,
        0.7421494,
        0.06945032,
        -0.02277208,
        0.8780185,
        0.76349473,
        1.0707821,
        0.424525,
        0.30425942,
        0.79608095,
    ]
)

KINOVA_EEF_TO_END = np.array(
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0.12], [0, 0, 0, 1]]
)

# NOTE: this is probably buggy - pushing for now but need to revise
T_franka_base_to_camera_3 = np.array([
    [-0.25634181, -0.91625915,  0.30782795, -0.34183448],
    [-0.78399233,  0.01081408, -0.62067631,  0.29268096],
    [ 0.56537147, -0.40044004, -0.72111225,  0.86914791],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
])