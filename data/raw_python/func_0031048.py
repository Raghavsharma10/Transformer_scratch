def get_fresh_primary_tumors(biospecimen):
    """Filter biospecimen data to only keep non-FFPE primary tumor samples.
    
    Parameters
    ----------
    biospecimen : `pandas.DataFrame`
        The biospecimen data frame. This type of data frame is returned by
        :meth:`get_biospecimen_data`.
    
    Returns
    -------
    `pandas.DataFrame`
        The filtered data frame.
    """
    df = biospecimen  # use shorter variable name

    # get rid of FFPE samples
    num_before = len(df.index)
    df = df.loc[~df['is_ffpe']]
    logger.info('Excluded %d files associated with FFPE samples '
                '(out of %d files in total).',
                num_before - len(df.index), num_before)

    # only keep primary tumors
    num_before = len(df.index)
    df = df.loc[df['sample_type'] == 'Primary Tumor']
    logger.info('Excluded %d files not corresponding to primary tumor '
                'samples (out of %d files in total).',
                num_before - len(df.index), num_before)

    return df