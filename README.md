# nqr-layerb: Pose-Dependent Signal Model for an Arm-Mounted NQR Coil

ASU MAE547 open project, J.F. Hansman

Computes coil fields, flip-angle maps, and relative NQR signal for a coil at a given pose over a simplified sample

## To Run Everything:
- Requires Python 3.11 or newer and an internet connection on the first run
- /cd to clone directory
  ```
  python run_checks.py
  ```
This creates a private environment in '.venv' inside this folder, installs the pinned packages from 'requirements.txt' into it, and runs all tests. Your own Python installation is not modified.

## Verification status:
| Check                                        | Measured  | Limit   | Result   | Note    |
| -------------------------------------------- | --------- | ------- | -------- | ------- |
| Check 1a: Loop field on axis vs textbook     | 6.03e-15  | 1.0e-03 | PASS     |         |
| Check 1b: Loop field near axis vs textbook   | 3.16e-12  | 1.0e-06 | PASS     |         |
| Check 2a: Loop field off-axis vs Maglibpy    | 3.26e-14  | 5.0e-03 | PASS     |         |
| Check 2b: Loop field off-axis vs Biot-Savart | 1.94e-13  | 5.0e-03 | PASS     |         |
| Check 3a: solenoid axis field = sum of loops | 1.25e-15  | 1.0e-12 | PASS     |         |
| Check 3b: (N=20) field near axis vs textbook | 1.99e-04  | 5.0e-03 | PASS     |         |
| Check 3b: (N=200) field near axis vs textbook| 2.00e-06  | 5.0e-03 | PASS     |         |
| Check 3c: convergence order, slope +2        | 4.49e-03  | 1.0e-01 | PASS     | fitted slope = -1.996 |
|rot_y: max abs(R^T*R -I)                      | 2.08e-17  | 1.0e-15 | PASS     |         |
|rot_y: abs(detR - 1)                          | 0.00e+00  | 1.0e-12 | PASS     |         |
|coil_rotation(-pi/2): axis vs. (0,0,-1)       | 1.22e-16  | 1.0e-15 | PASS     |         |
|frames: coil -> global -> coil round trip     | 1.73e-17  | 1.0e-14 | PASS     |         |
|Total voxel count vs. 40x32x20                | 0.00e+00  | 1.0e+00 | PASS     |         |
|Modeled slab's total vol. vs. 10x8x5cm        | 2.22e-16  | 1.0e-12 | PASS     |         |
|Top voxel's center vs. z = -1.25mm            | 0.00e+00  | 1.0e-15 | PASS     |         |
|Bottom voxel's center vs. z = -48.75mm)       | 0.00e+00  | 1.0e-15 | PASS     |         |
|lateral centroid offset amount (m)            | 5.56e-19  | 1.0e-15 | PASS     |         |
|Check 4a: f vs (sin a - acosa)/a^2            | 1.87e-14  | 1.0e-12 | PASS     |         |
|Check 4a: f vs. sqrt(pi/2a)J_3/2(a)           | 3.28e-15  | 1.0e-12 | PASS     |         |
|Check 4b: f maximum vs 0.4362                 | 1.82e-05  | 1.0e-03 | PASS     | max  0.4362
|Check 4b: f peak location vs 119.27 deg       | 4.48e-03  | 1.0e-01 | PASS     | peak at 119.27deg
|Check 4c: f zero vs 257.45deg                 | 3.40e-03  | 1.0e-01 | PASS     | minimum at  340.36deg
|Check 4c: f minimum location vs 340.4 deg     | 4.19e-02  | 1.0e-01 | PASS     | minimum at 340.36 deg
|Check 4c: f minimum vs -0.168                 | 7.00e-05  | 1.0e-03 | PASS     | minimum -0.1681
|Check 4d: MC powder average vs signal model   | 7.37e-01  | 4.0e+00 | PASS     | worst of 4 ex flip angles
|Reciprocity at r0 = [0.0, 0.0, 0.01]          | 2.22e-16  | 1.0e-02 | PASS     |         |
|Reciprocity at r0 = [0.015, 0.004, 0.008]     | 0.00e+00  | 1.0e-02 | PASS     |         |
|Reciprocity at r0 = [0.04, 0.0, -0.02]        | 8.88e-16  | 1.0e-02 | PASS     |         |
|Signal model: voxel at r_ref sees alpha_ref   | 00e+00    | 1.0e-12 | PASS     |         |
|Check 5: solenoid peak vs. 119.27 deg         |1.59e-02   | 2.0e+00 | PASS     | peak at  119.29 deg