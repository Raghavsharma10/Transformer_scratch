def run_list_error_summary(run_list, estimator_list, estimator_names,
                           n_simulate, **kwargs):
    """Wrapper which runs run_list_error_values then applies error_values
    summary to the resulting dataframe. See the docstrings for those two
    funcions for more details and for descriptions of parameters and output.
    """
    true_values = kwargs.pop('true_values', None)
    include_true_values = kwargs.pop('include_true_values', False)
    include_rmse = kwargs.pop('include_rmse', False)
    error_values = run_list_error_values(run_list, estimator_list,
                                         estimator_names, n_simulate, **kwargs)
    return error_values_summary(error_values, true_values=true_values,
                                include_true_values=include_true_values,
                                include_rmse=include_rmse)