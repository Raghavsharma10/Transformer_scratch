def check_anagrad(fun, x0, epsilon, threshold):
    """Check the analytical gradient using finite differences

       Arguments:
        | ``fun``  --  the function to be tested, more info below
        | ``x0``  --  the reference point around which the function should be
                      tested
        | ``epsilon``  --  a small scalar used for the finite differences
        | ``threshold``  --  the maximum acceptable difference between the
                             analytical gradient and the gradient obtained by
                             finite differentiation

       The function ``fun`` takes a mandatory argument ``x`` and an optional
       argument ``do_gradient``:
        | ``x``  --  the arguments of the function to be tested
        | ``do_gradient``  --  When False, only the function value is returned.
                               When True, a 2-tuple with the function value and
                               the gradient are returned [default=False]
    """
    N = len(x0)
    f0, ana_grad = fun(x0, do_gradient=True)
    for i in range(N):
        xh = x0.copy()
        xh[i] += 0.5*epsilon
        xl = x0.copy()
        xl[i] -= 0.5*epsilon
        num_grad_comp = (fun(xh)-fun(xl))/epsilon
        if abs(num_grad_comp - ana_grad[i]) > threshold:
            raise AssertionError("Error in the analytical gradient, component %i, got %s, should be about %s" % (i, ana_grad[i], num_grad_comp))