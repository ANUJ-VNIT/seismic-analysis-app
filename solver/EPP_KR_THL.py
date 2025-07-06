import numpy as np

def state_EPP(k, fy, fs_prev, u_old, u_new):
    """
    Elastic-Perfectly Plastic force update.
    """
    f_trial = fs_prev + k * (u_new - u_old)
    if abs(f_trial) > fy:
        return fy * np.sign(f_trial)
    else:
        return f_trial

def epp_kr_alpha_solver(m, ζ, Tn, Ry, accel, time, Rho=1.0):
    """
    Nonlinear EPP response of SDOF system using KR-alpha method.

    Returns:
    - normalized_u: Normalized displacement (u / uy)
    - normalized_fs: Normalized restoring force (fs / Fy)
    - time: Updated time array
    - ductility_demand
    - normalized_residual_deformation
    """

    # === Time Discretization ===
    dt = 0.001
    time_new = np.arange(time[0], time[-1], dt)
    accel_new = np.interp(time_new, time, accel)
    time_pad = np.arange(time_new[-1] + dt, time_new[-1] + 20 + dt, dt)
    accel_pad = np.zeros_like(time_pad)

    time = np.concatenate((time_new, time_pad))
    accel = np.concatenate((accel_new, accel_pad))
    n = len(time)

    # === System Properties ===
    k = (2 * np.pi / Tn) ** 2 * m
    c = 2 * ζ * np.sqrt(k * m)
    f = -m * accel * 9.81

    # === Step 1: Elastic Run to Determine Fy and uy ===
    u_lin = np.zeros(n)
    v_lin = np.zeros(n)
    a_lin = np.zeros(n)
    a_lin[0] = (f[0] - c * v_lin[0] - k * u_lin[0]) / m

    for i in range(n - 1):
        a_lin[i + 1] = (f[i + 1] - c * v_lin[i] - k * u_lin[i]) / m
        v_lin[i + 1] = v_lin[i] + dt * a_lin[i]
        u_lin[i + 1] = u_lin[i] + dt * v_lin[i] + 0.5 * dt**2 * a_lin[i]

    f0_max = np.max(np.abs(k * u_lin))
    Fy = f0_max / Ry
    uy = Fy / k

    # === KR-alpha Parameters ===
    alpha_m = (2 * Rho - 1) / (Rho + 1)
    alpha_f = Rho / (Rho + 1)
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (1 - alpha_m + alpha_f) ** 2

    Alpha = m + gamma * dt * c + beta * dt**2 * k
    Alpha1 = m / Alpha
    Alpha2 = (0.5 + gamma) * Alpha1
    Alpha3 = (alpha_m * m + alpha_f * gamma * dt * c + alpha_f * beta * dt**2 * k) / Alpha

    # === Preallocation ===
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    fs = np.zeros(n)

    # === Initial Conditions ===
    u[0] = 0.0
    v[0] = 0.0
    fs[0] = state_EPP(k, Fy, 0.0, 0.0, u[0])
    a[0] = (f[0] - c * v[0] - fs[0]) / m

    # === Time Integration ===
    for i in range(n - 1):
        # Step 1: Predict displacement and velocity
        v[i + 1] = v[i] + dt * Alpha1 * a[i]
        u[i + 1] = u[i] + dt * v[i] + dt**2 * Alpha2 * a[i]

        # Step 2: Update restoring force using EPP
        fs[i + 1] = state_EPP(k, Fy, fs[i], u[i], u[i + 1])

        # Step 3: KR-alpha evaluations
        v_alpha = (1 - alpha_f) * v[i + 1] + alpha_f * v[i]
        fs_alpha = (1 - alpha_f) * fs[i + 1] + alpha_f * fs[i]
        f_alpha = (1 - alpha_f) * f[i + 1] + alpha_f * f[i]

        # Step 4: Compute effective acceleration
        a_cap = (f_alpha - c * v_alpha - fs_alpha) / m
        a[i + 1] = (a_cap - Alpha3 * a[i]) / (1 - Alpha3)

    # === Final Correction for Resisting Force ===
    fs[-1] = min(Fy, max(-Fy, k * (u[-1] - u[-2])))

    # === Post-Processing ===
    normalized_u = u / uy
    normalized_fs = fs / Fy
    ductility_demand = np.max(np.abs(u)) / uy
    residual_deformation = abs(u[-1] - fs[-1] / k)
    normalized_residual_deformation = residual_deformation / uy

    return normalized_u, normalized_fs, time, ductility_demand, normalized_residual_deformation
