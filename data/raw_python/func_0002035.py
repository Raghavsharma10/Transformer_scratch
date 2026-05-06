def download_and_calibrate_parallel(list_of_ids, n=None):
    """Download and calibrate in parallel.

    Parameters
    ----------
    list_of_ids : list, optional
        container with img_ids to process
    n : int
        Number of cores for the parallel processing. Default: n_cores_system//2
    """
    setup_cluster(n_cores=n)
    c = Client()
    lbview = c.load_balanced_view()
    lbview.map_async(download_and_calibrate, list_of_ids)
    subprocess.Popen(["ipcluster", "stop", "--quiet"])