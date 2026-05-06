def combine_meta_data(files_dict, meta_data_v2=True):
    """
    Takes the dict of hdf5 files and combines their meta data tables into one new numpy record array.

    Parameters
    ----------
    meta_data_v2 : bool
        True for new (v2) meta data format, False for the old (v1) format.
    """
    if len(files_dict) > 10:
        logging.info("Combine the meta data from %d files", len(files_dict))
    # determine total length needed for the new combined array, thats the fastest way to combine arrays
    total_length = 0  # the total length of the new table
    for file_name in files_dict.iterkeys():
        with tb.open_file(file_name, mode="r") as in_file_h5:  # open the actual file
            total_length += in_file_h5.root.meta_data.shape[0]

    if meta_data_v2:
        meta_data_combined = np.empty((total_length, ), dtype=[
            ('index_start', np.uint32),
            ('index_stop', np.uint32),
            ('data_length', np.uint32),
            ('timestamp_start', np.float64),
            ('timestamp_stop', np.float64),
            ('error', np.uint32)])
    else:
        meta_data_combined = np.empty((total_length, ), dtype=[
            ('start_index', np.uint32),
            ('stop_index', np.uint32),
            ('length', np.uint32),
            ('timestamp', np.float64),
            ('error', np.uint32)])

    if len(files_dict) > 10:
        progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=total_length, term_width=80)
        progress_bar.start()

    index = 0

    # fill actual result array
    for file_name in files_dict.iterkeys():
        with tb.open_file(file_name, mode="r") as in_file_h5:  # open the actual file
            array_length = in_file_h5.root.meta_data.shape[0]
            meta_data_combined[index:index + array_length] = in_file_h5.root.meta_data[:]
            index += array_length
            if len(files_dict) > 10:
                progress_bar.update(index)
    if len(files_dict) > 10:
        progress_bar.finish()
    return meta_data_combined