def process_dynesty_run(results):
    """Transforms results from a dynesty run into the nestcheck dictionary
    format for analysis. This function has been tested with dynesty v9.2.0.

    Note that the nestcheck point weights and evidence will not be exactly
    the same as the dynesty ones as nestcheck calculates logX volumes more
    precisely (using the trapezium rule).

    This function does not require the birth_inds_given_contours and
    threads_given_birth_inds functions as dynesty results objects
    already include thread labels via their samples_id property. If the
    dynesty run is dynamic, the batch_bounds property is need to determine
    the threads' starting birth contours.

    Parameters
    ----------
    results: dynesty results object
        N.B. the remaining live points at termination must be included in the
        results (dynesty samplers' run_nested method does this if
        add_live_points=True - its default value).

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more details).
    """
    samples = np.zeros((results.samples.shape[0],
                        results.samples.shape[1] + 3))
    samples[:, 0] = results.logl
    samples[:, 1] = results.samples_id
    samples[:, 3:] = results.samples
    unique_th, first_inds = np.unique(results.samples_id, return_index=True)
    assert np.array_equal(unique_th, np.asarray(range(unique_th.shape[0])))
    thread_min_max = np.full((unique_th.shape[0], 2), np.nan)
    try:
        # Try processing standard nested sampling results
        assert unique_th.shape[0] == results.nlive
        assert np.array_equal(
            np.unique(results.samples_id[-results.nlive:]),
            np.asarray(range(results.nlive))), (
                'perhaps the final live points are not included?')
        thread_min_max[:, 0] = -np.inf
    except AttributeError:
        # If results has no nlive attribute, it must be dynamic nested sampling
        assert unique_th.shape[0] == sum(results.batch_nlive)
        for th_lab, ind in zip(unique_th, first_inds):
            thread_min_max[th_lab, 0] = (
                results.batch_bounds[results.samples_batch[ind], 0])
    for th_lab in unique_th:
        final_ind = np.where(results.samples_id == th_lab)[0][-1]
        thread_min_max[th_lab, 1] = results.logl[final_ind]
        samples[final_ind, 2] = -1
    assert np.all(~np.isnan(thread_min_max))
    run = nestcheck.ns_run_utils.dict_given_run_array(samples, thread_min_max)
    nestcheck.ns_run_utils.check_ns_run(run)
    return run