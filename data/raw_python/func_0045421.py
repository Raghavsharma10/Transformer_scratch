def numpy_compiler(model):
    """Take a triflow model and return optimized numpy routines.

    Parameters
    ----------
    model: triflow.Model:
        Model to compile

    Returns
    -------
    (numpy function, numpy function):
        Optimized routine that compute the evolution equations and their
        jacobian matrix.
    """

    def np_Min(args):
        a, b = args
        return np.where(a < b, a, b)

    def np_Max(args):
        a, b = args
        return np.where(a < b, b, a)

    def np_Heaviside(a):
        return np.where(a < 0, 1, 1)

    f_func = lambdify((model._symbolic_args),
                      expr=model.F_array.tolist(),
                      modules=[{"amax": np_Max,
                                "amin": np_Min,
                                "Heaviside": np_Heaviside},
                               "numpy"])

    j_func = lambdify((model._symbolic_args),
                      expr=model._J_sparse_array.tolist(),
                      modules=[{"amax": np_Max,
                                "amin": np_Min,
                                "Heaviside": np_Heaviside},
                               "numpy"])

    compute_F = partial(compute_F_numpy, model, f_func)
    compute_J = partial(compute_J_numpy, model, j_func)

    return compute_F, compute_J