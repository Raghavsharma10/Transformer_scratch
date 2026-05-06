def upload_metadata(directory, create_csv='', review='',
                    max_size='128m', verbose=False, debug=False):
    """
    Draft or review metadata files for uploading files to Open Humans.
    The target directory should either represent files for a single member (no
    subdirectories), or contain a subdirectory for each project member ID.

    :param directory: This field is the directory for which metadata has to be
        created.
    :param create_csv: This field is the output filepath to which csv file
        will be written.
    :param max_size: This field is the maximum file size. It's default value is
        None.
    :param verbose: This boolean field is the logging level. It's default value
        is False.
    :param debug: This boolean field is the logging level. It's default value
        is False.
    """
    set_log_level(debug, verbose)

    max_bytes = parse_size(max_size)
    if create_csv and review:
        raise ValueError("Either create_csv must be true or review must be " +
                         "true but not both")
    if review:
        if review_metadata_csv(directory, review):
            print("The metadata file has been reviewed and is valid.")
    elif create_csv:
        mk_metadata_csv(directory, create_csv, max_bytes=max_bytes)
    else:
        raise ValueError("Either create_csv must be true or review must be " +
                         "true but not both should be false")