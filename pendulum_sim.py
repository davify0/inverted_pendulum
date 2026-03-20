import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Pendulum parameters
g = 9.81  # gravity (m/s²)
L = 0.5   # length of pendulum (meters)
m = 0.1   # mass at the tip (kg)
b = 0.05  # damping (friction)

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

# Starting conditions
theta0 = 0.1
omega0 = 0.0
y0 = [theta0, omega0]

# Time setup
t_start = 0
t_end = 10
dt = 0.01
t_eval = np.arange(t_start, t_end, dt)

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

# Plot both side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

plt.tight_layout()
plt.show()