import numpy as np

def kr_alpha_linear_solver(m, ζ, Tn, accel, time, rho):
    """
    KR-alpha Method for SDOF system response to base excitation (acceleration input).

    Parameters:
    - m: Mass (kg)
    - ζ: Damping ratio (unitless)
    - Tn: Natural period of the system (s)
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
    a_hat = np.zeros(n)
    u[0] = 0
    v[0] = 0
    a_resp[0] = (f[0] - c * v[0] - k * u[0]) / m

    # KR-alpha parameters
    alpha_m = (2 * rho - 1) / (rho + 1)
    alpha_f = rho / (rho + 1)
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (1 - alpha_m + alpha_f) ** 2


    # Constants for the method
    alpha = m + gamma * dt * c + beta * dt ** 2 * k
    alpha1 = m / alpha
    alpha2 = ((0.5 + gamma) * m) / alpha
    alpha3 = (alpha_m * m + alpha_f * gamma * dt * c + alpha_f * beta * dt ** 2 * k) / alpha

    for i in range(n - 1):
        # Predict next velocity and displacement
        v[i + 1] = v[i] + dt * alpha1 * a_resp[i]
        u[i + 1] = u[i] + dt * v[i] + dt ** 2 * alpha2 * a_resp[i]

        # State determination
        fs_ip1 = k * u[i + 1]

        v_alpha = (1 - alpha_f) * v[i + 1] + alpha_f * v[i]
        fs_alpha = (1 - alpha_f) * fs_ip1 + alpha_f * k * u[i]
        p_alpha = (1 - alpha_f) * f[i + 1] + alpha_f * f[i]

        # 2.4 Compute predicted acceleration (a_hat)
        a_hat[i + 1] = (p_alpha - c * v_alpha - fs_alpha) / m

        # 2.5 Final acceleration update
        a_resp[i + 1] = (a_hat[i + 1] - alpha3 * a_resp[i]) / (1 - alpha3)

    return u, v, a_resp, time
