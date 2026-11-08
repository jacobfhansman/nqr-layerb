"""
Pose-dependent NQR signal model (Layer B)
"""
from .coils import (
    Coil, 
    Loop, 
    Solenoid, 
    loop_field_cyl, 
    loop_on_axis_Bz, 
    solenoid_center_Bz_sheet,
)

from .signal_model import(powder_response, received_signal, signal_contributions)
from .sample import(VoxelSample)
from .frames import(rot_y, rot_x, rot_z, coil_rotation, dh_transform, homogeneous, to_coil_frame)
from .arm import (nombase_xpos, joints, l1, l2, q_guess, CoilArm, tilt_axis, wrap)