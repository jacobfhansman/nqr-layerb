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

from .signal_model import(powder_response)
from .sample import(VoxelSample)