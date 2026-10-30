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