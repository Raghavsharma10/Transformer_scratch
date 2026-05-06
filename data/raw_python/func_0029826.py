def create_ramp_plan(err, ramp):
    """
    Formulate and execute on a plan to slowly add heat or cooling to the system

    `err` initial error (PV - SP)
    `ramp` the size of the ramp

    A ramp plan might yield MVs in this order at every timestep:
        [5, 0, 4, 0, 3, 0, 2, 0, 1]
        where err == 5 + 4 + 3 + 2 + 1
    """
    if ramp == 1:  # basecase
        yield int(err)
        while True:
            yield 0
    # np.arange(n).sum() == err
    # --> solve for n
    # err = (n - 1) * (n // 2) == .5 * n**2 - .5 * n
    # 0 = n**2 - n  --> solve for n
    n = np.abs(np.roots([.5, -.5, 0]).max())
    niter = int(ramp // (2 * n))  # 2 means add all MV in first half of ramp
    MV = n
    log.info('Initializing a ramp plan', extra=dict(
        ramp_size=ramp, err=err, niter=niter))
    for x in range(int(n)):
        budget = MV
        for x in range(niter):
            budget -= MV // niter
            yield int(np.sign(err) * (MV // niter))
        yield int(budget * np.sign(err))
        MV -= 1
    while True:
        yield 0