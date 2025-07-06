import numpy as np

def newmark_response_spectrum_solver(ζ, accel, time, gamma, beta):
    """
    Computes the Displacement Response Spectrum using Newmark-beta Method.

    Parameters:
    - ζ: Damping ratio (e.g. 0.02 for 2%)
    - accel: Ground acceleration array (in m/s²)
    - time: Time array (in seconds)
    - gamma, beta: Newmark integration parameters (default average acceleration method)

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
        ωn = 2 * np.pi / Tn
        k = ωn**2 * m
        c = 2 * ζ * np.sqrt(k * m)

        u = np.zeros(n)
        v = np.zeros(n)
        a = np.zeros(n)

        # Initial conditions
        u[0] = 0
        v[0] = 0
        a[0] = (f[0] - c * v[0] - k * u[0]) / m

        # Precompute constants
        a1 = m / (beta * dt**2) + c * gamma / (beta * dt)
        a2 = m / (beta * dt) + c * (gamma / beta - 1)
        a3 = m * (1 / (2 * beta) - 1) + c * dt * (gamma / (2 * beta) - 1)
        k_eff = k + a1

        for i in range(n - 1):
            rhs = f[i+1] + a1 * u[i] + a2 * v[i] + a3 * a[i]
            u[i+1] = rhs / k_eff
            v[i+1] = gamma / (beta * dt) * (u[i+1] - u[i]) + \
                     (1 - gamma / beta) * v[i] + dt * (1 - gamma / (2 * beta)) * a[i]
            a[i+1] = (u[i+1] - u[i]) / (beta * dt**2) - \
                     v[i] / (beta * dt) - (1 / (2 * beta) - 1) * a[i]

        max_disp[idx] = np.max(np.abs(u))

    return Tn_values, max_disp
