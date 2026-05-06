def corr_coeff(x1, x2, t, tau1, tau2):
    """Compute lagged correlation coefficient for two time series."""
    dt = t[1] - t[0]
    tau = np.arange(tau1, tau2+dt, dt)
    rho = np.zeros(len(tau))
    for n in range(len(tau)):
        i = np.abs(int(tau[n]/dt))
        if tau[n] >= 0: # Positive lag, push x2 forward in time
            seg2 = x2[0:-1-i]
            seg1 = x1[i:-1]
        elif tau[n] < 0: # Negative lag, push x2 back in time
            seg1 = x1[0:-i-1]
            seg2 = x2[i:-1]
        seg1 = seg1 - seg1.mean()
        seg2 = seg2 - seg2.mean()
        rho[n] = np.mean(seg1*seg2)/seg1.std()/seg2.std()
    return tau, rho