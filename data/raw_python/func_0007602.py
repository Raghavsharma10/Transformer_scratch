def upload_cli(directory, metadata_csv, master_token=None, member=None,
               access_token=None, safe=False, sync=False, max_size='128m',
               mode='default', verbose=False, debug=False):
    """
    Command line function for uploading files to OH.
    For more information visit
    :func:`upload<ohapi.command_line.upload>`.
    """
    return upload(directory, metadata_csv, master_token, member,
                  access_token, safe, sync, max_size,
                  mode, verbose, debug)