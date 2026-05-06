def run_thread_values(run, estimator_list):
    """Helper function for parallelising thread_values_df.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dictionary.
    estimator_list: list of functions

    Returns
    -------
    vals_array: numpy array
        Array of estimator values for each thread.
        Has shape (len(estimator_list), len(theads)).
    """
    threads = nestcheck.ns_run_utils.get_run_threads(run)
    vals_list = [nestcheck.ns_run_utils.run_estimators(th, estimator_list)
                 for th in threads]
    vals_array = np.stack(vals_list, axis=1)
    assert vals_array.shape == (len(estimator_list), len(threads))
    return vals_array