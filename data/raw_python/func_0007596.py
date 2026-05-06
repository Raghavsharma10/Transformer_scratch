def download_cli(directory, master_token=None, member=None, access_token=None,
                 source=None, project_data=False, max_size='128m',
                 verbose=False, debug=False, memberlist=None,
                 excludelist=None, id_filename=False):
    """
    Command line function for downloading data from project members to the
    target directory. For more information visit
    :func:`download<ohapi.command_line.download>`.
    """
    return download(directory, master_token, member, access_token, source,
                    project_data, max_size, verbose, debug, memberlist,
                    excludelist, id_filename)