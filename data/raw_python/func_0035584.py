def checkgrad(f, fprime, x, *args,**kw_args):
    """
    Analytical gradient calculation using a 3-point method

    """
    LG.debug("Checking gradient ...")
    import numpy as np

    # using machine precision to choose h
    eps = np.finfo(float).eps
    step = np.sqrt(eps)*(x.min())

    # shake things up a bit by taking random steps for each x dimension
    h = step*np.sign(np.random.uniform(-1, 1, x.size))

    f_ph = f(x+h, *args, **kw_args)
    f_mh = f(x-h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh)/(2*h)
    analytical_gradient = fprime(x, *args, **kw_args)
    ratio = (f_ph - f_mh)/(2*np.dot(h, analytical_gradient))

    h = np.zeros_like(x)
    for i in range(len(x)):
        pdb.set_trace()
    h[i] = step
    f_ph = f(x+h, *args, **kw_args)
    f_mh = f(x-h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh)/(2*step)
    analytical_gradient = fprime(x, *args, **kw_args)[i]
    ratio = (f_ph - f_mh)/(2*step*analytical_gradient)
    h[i] = 0
    LG.debug("[%d] numerical: %f, analytical: %f, ratio: %f" % (i, numerical_gradient,analytical_gradient,ratio))