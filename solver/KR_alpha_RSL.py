import numpy as np


def kr_alpha_response_spectrum_solver( ζ, accel, time, rho=1.0):
    """
    KR-alpha Method for SDOF system response to base excitation (acceleration input).

    Parameters:
    - m: Mass (kg)
    - ζ: Damping ratio (unitless)
    - Tn: Natural period of the system (s)
    - accel: Ground acceleration array (in m/s²)
    - time: Time array (same length as accel)
    - rho: KR-alpha parameter, default is 1.0

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
        a_hat = np.zeros(n)

        # KR-alpha parameters
        alpha_m = (2 * rho - 1) / (rho + 1)
        alpha_f = rho / (rho + 1)
        gamma = 0.5 - alpha_m + alpha_f
        beta = 0.25 * (1 - alpha_m + alpha_f) ** 2

        # Constants for the method
        alpha = m + gamma * dt * c + beta * dt ** 2 * k
        alpha1 = m / alpha
        alpha2 = ((0.5 + gamma) * m) / alpha
        alpha3 = (alpha_m * m + alpha_f * gamma * dt * c +
                  alpha_f * beta * dt ** 2 * k) / alpha
        for i in range(n - 1):
            # Predict next velocity and displacement
            v[i + 1] = v[i] + dt * alpha1 * a[i]
            u[i + 1] = u[i] + dt * v[i] + dt ** 2 * alpha2 * a[i]

            # State determination
            fs_ip1 = k * u[i + 1]

            v_alpha = (1 - alpha_f) * v[i + 1] + alpha_f * v[i]
            fs_alpha = (1 - alpha_f) * fs_ip1 + alpha_f * k * u[i]
            p_alpha = (1 - alpha_f) * f[i + 1] + alpha_f * f[i]

            # 2.4 Compute predicted acceleration (a_hat)
            a_hat[i + 1] = (p_alpha - c * v_alpha - fs_alpha) / m

            # 2.5 Final acceleration update
            a[i + 1] = (a_hat[i + 1] - alpha3 * a[i]) / (1 - alpha3)
        max_disp[idx] = np.max(np.abs(u))
    return Tn_values, max_disp
