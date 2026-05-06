def get_events_with_n_cluster(event_number, condition='n_cluster==1'):
    '''Selects the events with a certain number of cluster.

    Parameters
    ----------
    event_number : numpy.array

    Returns
    -------
    numpy.array
    '''

    logging.debug("Calculate events with clusters where " + condition)
    n_cluster_in_events = analysis_utils.get_n_cluster_in_events(event_number)
    n_cluster = n_cluster_in_events[:, 1]
#    return np.take(n_cluster_in_events, ne.evaluate(condition), axis=0)  # does not return 1d, bug?
    return n_cluster_in_events[ne.evaluate(condition), 0]