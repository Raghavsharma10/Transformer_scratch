def compute_fd_hessian(fun, x0, epsilon, anagrad=True):
    """Compute the Hessian using the finite difference method

       Arguments:
        | ``fun``  --  the function for which the Hessian should be computed,
                       more info below
        | ``x0``  --  the point at which the Hessian must be computed
        | ``epsilon``  --  a small scalar step size used to compute the finite
                           differences

       Optional argument:
        | ``anagrad``  --  when True, analytical gradients are used
                           [default=True]

       The function ``fun`` takes a mandatory argument ``x`` and an optional
       argument ``do_gradient``:
        | ``x``  --  the arguments of the function to be tested
        | ``do_gradient``  --  When False, only the function value is returned.
                               When True, a 2-tuple with the function value and
                               the gradient are returned [default=False]
    """
    N = len(x0)

    def compute_gradient(x):
        if anagrad:
            return fun(x, do_gradient=True)[1]
        else:
            gradient = np.zeros(N, float)
            for i in range(N):
                xh = x.copy()
                xh[i] += 0.5*epsilon
                xl = x.copy()
                xl[i] -= 0.5*epsilon
                gradient[i] = (fun(xh)-fun(xl))/epsilon
            return gradient

    hessian = np.zeros((N,N), float)
    for i in range(N):
        xh = x0.copy()
        xh[i] += 0.5*epsilon
        xl = x0.copy()
        xl[i] -= 0.5*epsilon
        hessian[i] = (compute_gradient(xh) - compute_gradient(xl))/epsilon

    return 0.5*(hessian + hessian.transpose())