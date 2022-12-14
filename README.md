# linopt_lib
This is a Linear Optimization Solver Library. The following Algorithm is implemented:
  - Two-phase Simplex Method

Testing performed against CVXPY LinProg solver as benchmark.

Run Instructions:

- step 1: clone https://github.com/bhargavg96/linopt_lib

- set python path to module linopt_lib

- change input problem in "linopt_driver.py". Possible keys : {'prob4', 'prob19', 'bjs36', 'bjs37', 'bjs44', 'bjs45', 'bjs411'}

- run linopt_driver.py