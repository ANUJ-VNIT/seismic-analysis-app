import numpy as np

def epp_newmark_solver(m, ζ, Tn, Ry, accel, time, gamma=0.5, beta=0.25):
    dt = 0.001
    time_new = np.arange(time[0], time[-1], dt)
    accel_new = np.interp(time_new, time, accel)

    time_pad = np.arange(time_new[-1] + dt, time_new[-1] + 20 + dt, dt)
    accel_pad = np.zeros_like(time_pad)
    time = np.concatenate((time_new, time_pad))
    accel = np.concatenate((accel_new, accel_pad))
    n = len(time)

    k = (2 * np.pi / Tn) ** 2 * m
    c = 2 * ζ * np.sqrt(k * m)
    f = -m * accel * 9.81

    # Linear elastic run
    u_lin = np.zeros(n)
    v_lin = np.zeros(n)
    a_lin = np.zeros(n)
    a_lin[0] = (f[0] - c * v_lin[0] - k * u_lin[0]) / m

    for i in range(n - 1):
        k_eff = m / (beta * dt**2) + gamma * c / (beta * dt) + k
        a_temp = m * (u_lin[i] / (beta * dt**2) + v_lin[i] / (beta * dt) + a_lin[i] * (1 / (2 * beta) - 1))
        c_temp = c * (u_lin[i] * gamma / (beta * dt) + v_lin[i] * (gamma / beta - 1) + dt * a_lin[i] * (gamma / (2 * beta) - 1))
        p_eff = f[i + 1] + a_temp + c_temp

        u_lin[i + 1] = p_eff / k_eff
        a_lin[i + 1] = (u_lin[i + 1] - u_lin[i]) / (beta * dt ** 2) - v_lin[i] / (beta * dt) - a_lin[i] * (1 / (2 * beta) - 1)
        v_lin[i + 1] = v_lin[i] + dt * ((1 - gamma) * a_lin[i] + gamma * a_lin[i + 1])

    f0_max = np.max(np.abs(k * u_lin))
    Fy = f0_max / Ry
    uy = Fy / k

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    fs = np.zeros(n)
    u_p = 0.0

    a[0] = (f[0] - c * v[0] - fs[0]) / m

    for i in range(n - 1):
        u_pred = u[i] + dt * v[i] + dt ** 2 * (0.5 - beta) * a[i]
        v_pred = v[i] + dt * (1 - gamma) * a[i]

        fs_trial = k * (u_pred - u_p)

        if abs(fs_trial) <= Fy:
            fs[i + 1] = fs_trial
        else:
            fs[i + 1] = Fy * np.sign(fs_trial)
            delta_u = u_pred - u[i]
            u_p += delta_u

        a[i + 1] = (f[i + 1] - c * v_pred - fs[i + 1]) / m

        u[i + 1] = u_pred + beta * dt ** 2 * a[i + 1]
        v[i + 1] = v_pred + gamma * dt * a[i + 1]

    fs[-1] = np.clip(k * (u[-1] - u_p), -Fy, Fy)

    normalized_u = u / uy
    normalized_fs = fs / Fy
    ductility_demand = np.max(np.abs(u)) / uy
    residual_deformation = abs(u[-1] - fs[-1] / k)
    normalized_residual_deformation = residual_deformation / uy

    return normalized_u, normalized_fs, time, ductility_demand, normalized_residual_deformation
