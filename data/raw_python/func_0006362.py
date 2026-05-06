def get_events_with_cluster_size(event_number, cluster_size, condition='cluster_size==1'):
    '''Selects the events with cluster of a given cluster size.

    Parameters
    ----------
    event_number : numpy.array
    cluster_size : numpy.array
    condition : string

    Returns
    -------
    numpy.array
    '''

    logging.debug("Calculate events with clusters with " + condition)
    return np.unique(event_number[ne.evaluate(condition)])