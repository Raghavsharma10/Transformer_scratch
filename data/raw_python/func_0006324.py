def histogram_cluster_table(analyzed_data_file, output_file, chunk_size=10000000):
    '''Reads in the cluster info table in chunks and histograms the seed pixels into one occupancy array.
    The 3rd dimension of the occupancy array is the number of different scan parameters used

    Parameters
    ----------
    analyzed_data_file : string
        HDF5 filename of the file containing the cluster table. If a scan parameter is given in the meta data, the occupancy histogramming is done per scan parameter step.

    Returns
    -------
    occupancy_array: numpy.array with dimensions (col, row, #scan_parameter)
    '''

    with tb.open_file(analyzed_data_file, mode="r") as in_file_h5:
        with tb.open_file(output_file, mode="w") as out_file_h5:
            histogram = PyDataHistograming()
            histogram.create_occupancy_hist(True)
            scan_parameters = None
            event_number_indices = None
            scan_parameter_indices = None
            try:
                meta_data = in_file_h5.root.meta_data[:]
                scan_parameters = analysis_utils.get_unique_scan_parameter_combinations(meta_data)
                if scan_parameters is not None:
                    scan_parameter_indices = np.array(range(0, len(scan_parameters)), dtype='u4')
                    event_number_indices = np.ascontiguousarray(scan_parameters['event_number']).astype(np.uint64)
                    histogram.add_meta_event_index(event_number_indices, array_length=len(scan_parameters['event_number']))
                    histogram.add_scan_parameter(scan_parameter_indices)
                    logging.info("Add %d different scan parameter(s) for analysis", len(scan_parameters))
                else:
                    logging.info("No scan parameter data provided")
                    histogram.set_no_scan_parameter()
            except tb.exceptions.NoSuchNodeError:
                logging.info("No meta data provided, use no scan parameter")
                histogram.set_no_scan_parameter()

            logging.info('Histogram cluster seeds...')
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Cluster.shape[0], term_width=80)
            progress_bar.start()
            total_cluster = 0  # to check analysis
            for cluster, index in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, chunk_size=chunk_size):
                total_cluster += len(cluster)
                histogram.add_cluster_seed_hits(cluster, len(cluster))
                progress_bar.update(index)
            progress_bar.finish()

            filter_table = tb.Filters(complib='blosc', complevel=5, fletcher32=False)  # compression of the written data
            occupancy_array = histogram.get_occupancy().T
            occupancy_array_table = out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram', atom=tb.Atom.from_dtype(occupancy_array.dtype), shape=occupancy_array.shape, filters=filter_table)
            occupancy_array_table[:] = occupancy_array

            if total_cluster != np.sum(occupancy_array):
                logging.warning('Analysis shows inconsistent number of cluster used. Check needed!')
            in_file_h5.root.meta_data.copy(out_file_h5.root)