import numpy as np
from scipy.interpolate import interp1d


def cd_response_spectrum_solver(ζ, accel, time):
    """
    Computes the Displacement Response Spectrum using Central Difference Method (CDM).

    Parameters:
    - accel: Ground acceleration array (in m/s²)
    - time: Time array (in seconds)
    - damping_ratio: ζ (default 2%)
    - m: Mass of the system (default 1.0 kg)
    - dt: Time step to interpolate signal (default 0.001s)
    - Tn_max: Maximum Time Period for RS (default 3s)
    - Tn_step: Resolution of Time Periods (default 0.01s)

    Returns:
    - Tn_values: Array of natural periods
    - max_disp: Array of max displacements for each Tn (in inches)
    """

    # Resample time and interpolate acceleration
    time_new = np.array(time)
    accel_new = np.array(accel)
    dt = time_new[1] - time_new[0]  # Time step
    m = 1.0  # Mass in kg
    damping_ratio = ζ  # Damping ratio

    n = len(time_new)  # number of time steps
    f = -m * accel_new  # force in N

    Tn_values = np.arange(0.01, 3, 0.01)
    max_disp = np.zeros(len(Tn_values))

    for Tn in Tn_values:

        # ➤ Central Difference Method Stability Check
        ωn = 2 * np.pi / Tn
        if dt >= 2 / ωn:
            max_disp[Tn_values == Tn] = np.nan
            continue

        k = (2 * np.pi / Tn) ** 2 * m  # spring constant in N/m
        c = 2 * ζ * np.sqrt(k * m)  # damping coefficient in Ns/m

        u = np.zeros(n)  # displacement in m
        v = np.zeros(n)  # velocity in m/s
        a = np.zeros(n)  # acceleration in m/s^2
        u[0] = 0  # initial displacement
        v[0] = 0  # initial velocity
        a[0] = (f[0] - c * v[0] - k * u[0]) / m  # initial acceleration
        u_minus_1 = u[0] - dt * v[0] + (dt**2/2)*a[0]  # Displacement at t-1

        k_hat = m/dt**2 + c/(2*dt)
        a1 = m/dt**2 - c/(2*dt)
        b = k - 2*m/dt**2

        # main loop for displacement, velocity, and acceleration
        for i in range(n-1):
            if i == 0:
                u[i+1] = (f[i] - a1*u_minus_1 - b*u[i]) / k_hat
                v[i+1] = (u[i+1]-u_minus_1) / (2*dt)
                a[i+1] = (u[i+1] - 2*u[i] + u_minus_1) / (dt**2)
            else:
                u[i+1] = (f[i] - a1*u[i-1] - b*u[i]) / k_hat
                v[i+1] = (u[i+1]-u[i-1]) / (2*dt)
                a[i+1] = (u[i+1] - 2*u[i] + u[i-1]) / (dt**2)

        max_disp[Tn_values == Tn] = np.max(u)

    return Tn_values, max_disp
