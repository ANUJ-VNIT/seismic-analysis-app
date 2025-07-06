import numpy as np

def newmark_solver(m, ζ, Tn, accel, time, gamma, beta):
    """
    Newmark-beta Method for SDOF system response to base excitation.

    Parameters:
    - m: Mass (kg)
    - c: Damping coefficient (Ns/m)
    - Tn: Natural period (s)
    - accel: Ground acceleration array (in m/s^2)
    - time: Time array (same length as accel)
    - gamma: Newmark parameter 
    - beta: Newmark parameter 

    Returns:
    - u: Displacement (m)
    - v: Velocity (m/s)
    - a: Acceleration (m/s^2)
    - t: Time array (s)
    """
    accel = np.array(accel)
    time = np.array(time)
    dt = time[1] - time[0]
    n = len(time)

    
    k = (2 * np.pi / Tn)**2 * m  # Convert Time period to stiffness
    c = 2 * ζ * np.sqrt(k * m)  # Damping coefficient
    f = -m * accel  # base excitation force

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    u[0] = 0
    v[0] = 0
    a[0] = (f[0] - c * v[0] - k * u[0]) / m

    # Newmark constants
    a1 = m / (beta * dt**2) + c * gamma / (beta * dt)
    a2 = m / (beta * dt) + c * (gamma / beta - 1)
    a3 = m / (2 * beta) - m + dt * c * (gamma / (2 * beta) - 1)
    k_hat = k + a1

    for i in range(n - 1):
        rhs = f[i+1] + a1 * u[i] + a2 * v[i] + a3 * a[i]
        u[i+1] = rhs / k_hat
        v[i+1] = gamma / (beta * dt) * (u[i+1] - u[i]) + (1 - gamma / beta) * v[i] + dt * (1 - gamma / (2 * beta)) * a[i]
        a[i+1] = (u[i+1] - u[i]) / (beta * dt**2) - v[i] / (beta * dt) - (1 / (2 * beta) - 1) * a[i]

    return u, v, a, time
