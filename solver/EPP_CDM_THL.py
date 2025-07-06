import numpy as np


def epp_time_history_solver(m, ζ, Tn, Ry, accel, time):
    """
    Elastic-Perfectly Plastic (EPP) response of SDOF system using Central Difference Method.

    Returns:
    - normalized_u_epp: Normalized displacement array (u/uy)
    - normalized_f_s: Normalized restoring force array (f_s/Fy)
    - time: Time array (in seconds)
    """
    dt = 0.001  # time step in seconds
    time_new = np.arange(time[0], time[-1], dt)
    accel_new = np.interp(time_new, time, accel)
    time_pad = np.arange(time_new[-1] + dt, time_new[-1] + 20 + dt, dt)
    accel_pad = np.zeros(len(time_pad))
    time = np.concatenate((time_new, time_pad))
    accel = np.concatenate((accel_new, accel_pad))

    n = len(time)

    k = (2 * np.pi / Tn) ** 2 * m  # spring constant
    c = 2 * ζ * np.sqrt(k * m)  # damping coefficient
    f = -m * accel * 9.81  # excitation force in N

    # Step 1: Linear elastic run to get peak force
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    u[0] = 0
    v[0] = 0
    a[0] = (f[0] - c * v[0] - k * u[0]) / m
    u_minus_1 = u[0] - dt * v[0] + 0.5 * dt**2 * a[0]

    k_bar = m / dt**2 + c / (2 * dt)
    a1 = m / dt**2 - c / (2 * dt)
    b = 2 * m / dt**2

    for i in range(n - 1):
        if i == 0:
            p_bar = f[i] - a1 * u_minus_1 + b * u[i]
            u[i + 1] = p_bar / k_bar
            v[i] = (u[i + 1] - u_minus_1) / (2 * dt)
            a[i] = (u[i + 1] - 2 * u[i] + u_minus_1) / dt**2
        else:
            p_bar = f[i] - a1 * u[i - 1] + b * u[i]
            u[i + 1] = p_bar / k_bar
            v[i] = (u[i + 1] - u[i - 1]) / (2 * dt)
            a[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dt**2

    f0_max = np.max(np.abs(k * u))
    Fy = f0_max / Ry
    uy = Fy / k

    # Step 2: EPP analysis
    u_epp = np.zeros(n)
    v_epp = np.zeros(n)
    a_epp = np.zeros(n)
    f_s = np.zeros(n)

    u_epp[0] = 0
    v_epp[0] = 0
    a_epp[0] = (f[0] - c * v_epp[0] - k * u_epp[0]) / m
    u_minus_1_epp = u_epp[0] - dt * v_epp[0] + 0.5 * dt**2 * a_epp[0]

    k_bar = m / dt**2 + c / (2 * dt)
    a1 = m / dt**2 - c / (2 * dt)
    b = 2 * m / dt**2

    # Initial restoring force
    def state_epp(k, fy, fsi, u1, u2):
        delta_u = u2 - u1
        fs_trial = fsi + k * delta_u
        if abs(fs_trial) > fy:
            return fy * np.sign(fs_trial)
        else:
            return fs_trial

    f_s[0] = state_epp(k, Fy, 0, 0, u_epp[0])

    for i in range(n - 1):
        if i == 0:
            p_bar = f[i] - a1 * u_minus_1_epp - f_s[i] + b * u_epp[i]
            u_epp[i + 1] = p_bar / k_bar
            v_epp[i] = (u_epp[i + 1] - u_minus_1_epp) / (2 * dt)
            a_epp[i] = (u_epp[i + 1] - 2 * u_epp[i] + u_minus_1_epp) / dt**2
            f_s[i + 1] = state_epp(k, Fy, f_s[i], u_epp[i], u_epp[i + 1])
        else:
            p_bar = f[i] - a1 * u_epp[i - 1] - f_s[i] + b * u_epp[i]
            u_epp[i + 1] = p_bar / k_bar
            v_epp[i] = (u_epp[i + 1] - u_epp[i - 1]) / (2 * dt)
            a_epp[i] = (u_epp[i + 1] - 2 * u_epp[i] + u_epp[i - 1]) / dt**2
            f_s[i + 1] = state_epp(k, Fy, f_s[i], u_epp[i], u_epp[i + 1])

    normalized_u_epp = u_epp / uy
    normalized_f_s = f_s / Fy
    u_max = np.max(np.abs(u_epp))
    ductility_demand = u_max / uy
    residual_deformation = abs(u_epp[-1] - f_s[-1] / k)
    normalized_residual_deformation = residual_deformation / uy

    return normalized_u_epp, normalized_f_s, time, ductility_demand, normalized_residual_deformation
