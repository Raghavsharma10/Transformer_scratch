def get_n_cluster_per_event_hist(cluster_table):
    '''Calculates the number of cluster in every event.

    Parameters
    ----------
    cluster_table : pytables.table

    Returns
    -------
    numpy.Histogram
    '''
    logging.info("Histogram number of cluster per event")
    cluster_in_events = analysis_utils.get_n_cluster_in_events(cluster_table)[:, 1]  # get the number of cluster for every event
    return np.histogram(cluster_in_events, bins=range(0, np.max(cluster_in_events) + 2))