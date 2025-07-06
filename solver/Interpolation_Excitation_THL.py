import numpy as np

def interpolation_excitation_solver(m, ζ, Tn, accel, time):
    """
    Interpolation Excitation Method for SDOF system response to base excitation (acceleration input).

    Parameters:
    - m: Mass (kg)
    - ζ: Damping ratio (unitless)
    - Tn: Natural period (s)
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
    wn = 2 * np.pi / Tn  # Natural frequency (rad/s)
    wd = wn * np.sqrt(1 - ζ**2)  # Damped natural frequency (rad/s)
    c = 2 * ζ * m * wn  # Damping coefficient from damping ratio
    f = -m * accel  # base excitation force

    u = np.zeros(n)
    v = np.zeros(n)
    
    
    


    exp_term = np.exp(-ζ * wn * dt)

    A = exp_term * (np.cos(wd * dt) + (ζ / np.sqrt(1 - ζ**2)) * np.sin(wd * dt))
    B = exp_term * (np.sin(wd * dt) / wd)

    C = (1 / k) * (
        (2 * ζ / (wn * dt)) +
        exp_term * (
            ((1 - 2 * ζ**2) / (wd * dt) - (ζ / np.sqrt(1 - ζ**2))) * np.sin(wd * dt) -
            (1 + (2 * ζ) / (wn * dt)) * np.cos(wd * dt)
        )
    )

    D = (1 / k) * (
        1 - (2 * ζ) / (wn * dt) +
        exp_term * (
            ((2 * ζ**2 - 1) / (wd * dt)) * np.sin(wd * dt) +
            (2 * ζ / (wn * dt)) * np.cos(wd * dt)
        )
    )

    A_dash = -exp_term * ((wn / np.sqrt(1 - ζ**2)) * np.sin(wd * dt))

    B_dash = exp_term * (
        np.cos(wd * dt) - (ζ / np.sqrt(1 - ζ**2)) * np.sin(wd * dt)
    )

    C_dash = (1 / k) * (
        -1 / dt +
        exp_term * (
            ((wn / np.sqrt(1 - ζ**2)) + (ζ / (dt * np.sqrt(1 - ζ**2)))) * np.sin(wd * dt) +
            (1 / dt) * np.cos(wd * dt)
        )
    )

    D_dash = (1 / (k * dt)) * (
        1 - exp_term * (
            (ζ / np.sqrt(1 - ζ**2)) * np.sin(wd * dt) +
            np.cos(wd * dt)
        )
    )


    u[0] = 0
    v[0] = 0
    


    for i in range(n-1):
        u[i+1] = A * u[i] + B * v[i] + C * f[i] + D * f[i+1]
        v[i+1] = A_dash * u[i] + B_dash * v[i] + C_dash * f[i] + D_dash * f[i+1]

    return u,v,time    
        