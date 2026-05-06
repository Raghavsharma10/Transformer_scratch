def upload_metadata_cli(directory, create_csv='', review='',
                        max_size='128m', verbose=False, debug=False):
    """
    Command line function for drafting or reviewing metadata files.
    For more information visit
    :func:`upload_metadata<ohapi.command_line.upload_metadata>`.
    """
    return upload_metadata(directory, create_csv, review,
                           max_size, verbose, debug)