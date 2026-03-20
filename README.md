# Inverted Pendulum Control Research

Undergraduate research project investigating balance control strategies 
for an inverted pendulum system. Built as part of a self-directed research 
initiative toward IEEE Region 8 Student Paper Contest submission.

## Project Overview
The goal is to design, simulate, and physically implement two control 
strategies — PID and LQR — and rigorously compare their performance. 
The physical robot will be used to validate simulation results and 
analyze the sim-to-real gap.

## Current Progress
- Uncontrolled pendulum simulation complete
- PID controller implemented, tuned and tested
- LQR optimal controller implemented and tested
- Three-way comparison visualization complete
- Performance metrics quantified (settling time, overshoot, steady state error)
- Disturbance rejection test complete — LQR outperforms PID under external disturbance

## Tools and Environment
- Python 3.12
- numpy, scipy, matplotlib
- ESP32 microcontroller (hardware phase)
- MPU6050 IMU sensor (hardware phase)

## Roadmap
- [x] Simulation: uncontrolled pendulum
- [x] Simulation: PID controller
- [x] Simulation: LQR controller
- [x] Quantified performance metrics
- [x] Disturbance rejection testing
- [ ] Physical robot build
- [ ] Sim-to-real comparison
- [ ] IEEE paper writeup

## Author
Onyia Ifeanyi David, Mechatronics Engineering student, Pan-Atlantic University, Lagos, Nigeria.
Year 2 of 5.
