def read_ring_images_index():
    """Filter cumulative index for ring images.

    This is done by matching the column TARGET_DESC to contain the string 'ring'

    Returns
    -------
    pandas.DataFrame
        data table containing only meta-data for ring images
    """
    meta = read_cumulative_iss_index()
    ringfilter = meta.TARGET_DESC.str.contains("ring", case=False)
    return meta[ringfilter]