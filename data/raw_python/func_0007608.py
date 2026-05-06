def public_data_download_cli(source, username, directory, max_size, quiet,
                             debug):
    """
    Command line tools for downloading public data.
    """
    return public_download(source, username, directory, max_size, quiet, debug)