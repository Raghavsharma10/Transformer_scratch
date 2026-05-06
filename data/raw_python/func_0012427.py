def get_latex_name(func_in, **kwargs):
    """
    Produce a latex formatted name for each function for use in labelling
    results.

    Parameters
    ----------
    func_in: function
    kwargs: dict, optional
        Kwargs for function.

    Returns
    -------
    latex_name: str
        Latex formatted name for the function.
    """
    if isinstance(func_in, functools.partial):
        func = func_in.func
        assert not set(func_in.keywords) & set(kwargs), (
            'kwargs={0} and func_in.keywords={1} contain repeated keys'
            .format(kwargs, func_in.keywords))
        kwargs.update(func_in.keywords)
    else:
        func = func_in
    param_ind = kwargs.pop('param_ind', 0)
    probability = kwargs.pop('probability', 0.5)
    kwargs.pop('handle_indexerror', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    ind_str = r'{\hat{' + str(param_ind + 1) + '}}'
    latex_name_dict = {
        'count_samples': r'samples',
        'logz': r'$\mathrm{log} \mathcal{Z}$',
        'evidence': r'$\mathcal{Z}$',
        'r_mean': r'$\overline{|\theta|}$',
        'param_mean': r'$\overline{\theta_' + ind_str + '}$',
        'param_squared_mean': r'$\overline{\theta^2_' + ind_str + '}$'}
    # Add credible interval names
    if probability == 0.5:
        cred_str = r'$\mathrm{median}('
    else:
        # format percent without trailing zeros
        percent_str = ('%f' % (probability * 100)).rstrip('0').rstrip('.')
        cred_str = r'$\mathrm{C.I.}_{' + percent_str + r'\%}('
    latex_name_dict['param_cred'] = cred_str + r'\theta_' + ind_str + ')$'
    latex_name_dict['r_cred'] = cred_str + r'|\theta|)$'
    try:
        return latex_name_dict[func.__name__]
    except KeyError as err:
        err.args = err.args + ('get_latex_name not yet set up for ' +
                               func.__name__,)
        raise