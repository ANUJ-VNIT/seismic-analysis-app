import numpy as np

def interpolation_response_spectrum_solver(ζ, accel, time):
    """
    Computes the Displacement Response Spectrum using Interpolation Excitation Method.

    Parameters:
    - ζ: Damping ratio (e.g. 0.02 for 2%)
    - accel: Ground acceleration array (in m/s²)
    - time: Time array (in seconds)

    Returns:
    - Tn_values: Array of natural periods
    - max_disp: Array of max displacements for each Tn (in meters)
    """
    
    time = np.array(time)
    accel = np.array(accel)
    dt = time[1] - time[0]
    m = 1.0  # Mass in kg
    f = -m * accel  # base excitation force
    n = len(time)

    Tn_values = np.arange(0.01, 3.0, 0.01)  # periods from 0.01 to 3s
    max_disp = np.zeros(len(Tn_values))

    for idx, Tn in enumerate(Tn_values):
        wn = 2 * np.pi / Tn
        wd = wn * np.sqrt(1 - ζ**2)
        k = wn**2 * m
        c = 2 * ζ * np.sqrt(k * m)

        u = np.zeros(n)
        v = np.zeros(n)
        

        # Initial conditions
        u[0] = 0
        v[0] = 0
       

       # Pre compute constants

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
        for i in range(n - 1):
            u[i + 1] = A * u[i] + B * v[i] + C * f[i] + D * f[i + 1]
            v[i + 1] = A_dash * u[i] + B_dash * v[i] + C_dash * f[i] + D_dash * f[i + 1]
        max_disp[idx] = np.max(np.abs(u))
    return Tn_values, max_disp


