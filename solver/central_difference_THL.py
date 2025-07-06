import numpy as np


def central_difference_solver(m, ζ, Tn, accel, time):
    """
    Central Difference Method for SDOF system response to base excitation (acceleration input).

    Parameters:
    - m: Mass (kg)
    - c: Damping coefficient (Ns/m)
    - k: Stiffness (N/m)
    - accel: Ground acceleration array (in m/s^2)
    - time: Time array (same length as accel)

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
    c = 2 * ζ * np.sqrt(k * m)  # Damping coefficient from damping ratio
    f = -m * accel  # base excitation force

    u = np.zeros(n)
    v = np.zeros(n)
    a_resp = np.zeros(n)
    u[0] = 0
    v[0] = 0
    a_resp[0] = (f[0] - c * v[0] - k * u[0]) / m
    u_minus_1 = u[0] - dt * v[0] + (dt**2 / 2) * a_resp[0]

    k_hat = m / dt**2 + c / (2 * dt)
    a1 = m / dt**2 - c / (2 * dt)
    b = k - 2 * m / dt**2

    for i in range(n - 1):
        if i == 0:
            u[i+1] = (f[i] - a1*u_minus_1 - b*u[i]) / k_hat
            v[i+1] = (u[i+1]-u_minus_1) / (2*dt)
            a_resp[i+1] = (u[i+1] - 2*u[i] + u_minus_1) / (dt**2)
        else:
            u[i+1] = (f[i] - a1*u[i-1] - b*u[i]) / k_hat
            v[i+1] = (u[i+1]-u[i-1]) / (2*dt)
            a_resp[i+1] = (u[i+1] - 2*u[i] + u[i-1]) / (dt**2)

    return u, v, a_resp, time
