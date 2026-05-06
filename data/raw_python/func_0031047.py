def get_unique_sample_files(file_samples):
    """Filter file_sample data frame to only keep one file per sample.
    
    Params
    ------
    file_samples : `pandas.DataFrame`
        A data frame containing a mapping between file IDs and sample barcodes.
        This type of data frame is returned by :meth:`get_file_samples`.
        
    Returns
    -------
    `pandas.DataFrame`
        The filtered data frame.

    Notes
    -----
    In order to remove redundant files in a consistent fashion, the samples are
    sorted by file ID, and then the first file for each sample is kept.
    """
    assert isinstance(file_samples, pd.DataFrame)

    df = file_samples  # use shorter variable name

    # sort data frame by file ID
    df = df.sort_values('file_id')

    # - some samples have multiple files with the same barcode,
    #   corresponding to different aliquots
    # get rid of those duplicates
    logger.info('Original number of files: %d', len(df.index))
    df.drop_duplicates('sample_barcode', keep='first', inplace=True)
    logger.info('Number of files after removing duplicates from different '
                'aliquots: %d', len(df.index))

    # - some samples also have multiple files corresponding to different vials
    # add an auxilliary column that contains the sample barcode without the
    # vial tag (first 15 characters)
    df['sample_barcode15'] = df['sample_barcode'].apply(lambda x: x[:15])

    # use auxilliary column to get rid of duplicate files
    df.drop_duplicates('sample_barcode15', keep='first', inplace=True)
    logger.info('Number of files after removing duplicates from different '
                'vials: %d', len(df.index))

    # get rid of auxilliary column 
    df.drop('sample_barcode15', axis=1, inplace=True)

    # restore original order using the (numerical) index
    df.sort_index(inplace=True)
    
    # return the filtered data frame
    return df