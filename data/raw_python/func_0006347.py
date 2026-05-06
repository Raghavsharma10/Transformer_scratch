def get_data_file_names_from_scan_base(scan_base, filter_str=['_analyzed.h5', '_interpreted.h5', '_cut.h5', '_result.h5', '_hists.h5'], sort_by_time=True, meta_data_v2=True):
    """
    Generate a list of .h5 files which have a similar file name.

    Parameters
    ----------
    scan_base : list, string
        List of string or string of the scan base names. The scan_base will be used to search for files containing the string. The .h5 file extension will be added automatically.
    filter : list, string
        List of string or string which are used to filter the returned filenames. File names containing filter_str in the file name will not be returned. Use None to disable filter.
    sort_by_time : bool
        If True, return file name list sorted from oldest to newest. The time from meta table will be used to sort the files.
    meta_data_v2 : bool
        True for new (v2) meta data format, False for the old (v1) format.

    Returns
    -------
    data_files : list
        List of file names matching the obove conditions.
    """
    data_files = []
    if scan_base is None:
        return data_files
    if isinstance(scan_base, basestring):
        scan_base = [scan_base]
    for scan_base_str in scan_base:
        if '.h5' == os.path.splitext(scan_base_str)[1]:
            data_files.append(scan_base_str)
        else:
            data_files.extend(glob.glob(scan_base_str + '*.h5'))

    if filter_str:
        if isinstance(filter_str, basestring):
            filter_str = [filter_str]
        data_files = filter(lambda data_file: not any([(True if x in data_file else False) for x in filter_str]), data_files)
    if sort_by_time and len(data_files) > 1:
        f_list = {}
        for data_file in data_files:
            with tb.open_file(data_file, mode="r") as h5_file:
                try:
                    meta_data = h5_file.root.meta_data
                except tb.NoSuchNodeError:
                    logging.warning("File %s is missing meta_data" % h5_file.filename)
                else:
                    try:
                        if meta_data_v2:
                            timestamp = meta_data[0]["timestamp_start"]
                        else:
                            timestamp = meta_data[0]["timestamp"]
                    except IndexError:
                        logging.info("File %s has empty meta_data" % h5_file.filename)
                    else:
                        f_list[data_file] = timestamp

        data_files = list(sorted(f_list, key=f_list.__getitem__, reverse=False))
    return data_files