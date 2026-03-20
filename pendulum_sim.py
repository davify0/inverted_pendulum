import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# Pendulum parameters
g = 9.81  # gravity (m/s²)
L = 0.5   # length of pendulum (meters)
m = 0.1   # mass at the tip (kg)
b = 0.05  # damping (friction)

# Starting conditions
theta0 = 0.1
omega0 = 0.0
y0 = [theta0, omega0]

# Time setup
t_start = 0
t_end = 10
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

# Uncontrolled pendulum
def pendulum(t, y):
    theta = y[0]
    omega = y[1]
    dtheta = omega
    domega = -(g/L)*np.sin(theta) - (b/m)*omega
    return [dtheta, domega]

# PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = error
        return output

# Pendulum with PID control
def pendulum_pid(t, y, pid, dt):
    theta = y[0]
    omega = y[1]
    error = 0 - theta
    force = pid.compute(error, dt)
    dtheta = omega
    domega = -(g/L)*np.sin(theta) - (b/m)*omega + (force/m*L)
    return [dtheta, domega]

# LQR Controller
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

# System matrices
A = np.array([[0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, (g/L), 0]])

B = np.array([[0],
              [1/m],
              [0],
              [1/(m*L)]])

# Weighting matrices
Q = np.diag([1, 1, 10, 1])
R = np.array([[0.1]])

# Solve for K
K = lqr(A, B, Q, R)
print("LQR Gain K:", K)

# Pendulum with LQR control
def pendulum_lqr(t, y):
    theta = y[0]
    omega = y[1]
    state = np.array([0, 0, theta, omega])
    force = -K @ state
    force = float(force)
    dtheta = omega
    domega = -(g/L)*np.sin(theta) - (b/m)*omega + (force/m*L)
    return [dtheta, domega]

# Uncontrolled simulation
solution_uncontrolled = solve_ivp(pendulum, (t_start, t_end), y0, t_eval=t_eval)
t_unc = solution_uncontrolled.t
theta_unc = solution_uncontrolled.y[0]

# PID simulation
pid = PIDController(Kp=50, Ki=1, Kd=10)
solution_pid = solve_ivp(
    lambda t, y: pendulum_pid(t, y, pid, dt),
    (t_start, t_end),
    y0,
    t_eval=t_eval
)
t_pid = solution_pid.t
theta_pid = solution_pid.y[0]

# LQR simulation
solution_lqr = solve_ivp(pendulum_lqr, (t_start, t_end), y0, t_eval=t_eval)
t_lqr = solution_lqr.t
theta_lqr = solution_lqr.y[0]

# Plot all three
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.plot(t_unc, theta_unc, color='blue', label='Uncontrolled')
ax1.axhline(y=0, color='black', linestyle='--', label='Target')
ax1.set_title('No Controller')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Angle (radians)')
ax1.legend()
ax1.grid(True)

ax2.plot(t_pid, theta_pid, color='red', label='PID controlled')
ax2.axhline(y=0, color='black', linestyle='--', label='Target')
ax2.set_title('With PID Controller')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Angle (radians)')
ax2.legend()
ax2.grid(True)

ax3.plot(t_lqr, theta_lqr, color='green', label='LQR controlled')
ax3.axhline(y=0, color='black', linestyle='--', label='Target')
ax3.set_title('With LQR Controller')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Angle (radians)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()