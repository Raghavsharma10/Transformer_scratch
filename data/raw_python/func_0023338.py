def _fetch_file(url, file_name, print_destination=True):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    url: string
        The url of file to be downloaded.
    file_name: string
        Name, along with the path, of where downloaded file will be saved.
    print_destination: bool, optional
        If true, destination of where file was saved will be printed after
        download finishes.
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py

    temp_file_name = file_name + ".part"
    local_file = None
    initial_size = 0
    # Checking file size and displaying it alongside the download url
    n_try = 3
    for ii in range(n_try):
        try:
            data = urllib.request.urlopen(url, timeout=15.)
        except Exception as e:
            if ii == n_try - 1:
                raise RuntimeError('Error while fetching file %s.\n'
                                   'Dataset fetching aborted (%s)' % (url, e))
    try:
        file_size = int(data.headers['Content-Length'].strip())
        print('Downloading data from %s (%s)' % (url, sizeof_fmt(file_size)))
        local_file = open(temp_file_name, "wb")
        _chunk_read(data, local_file, initial_size=initial_size)
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_file_name, file_name)
        if print_destination is True:
            sys.stdout.write('File saved as %s.\n' % file_name)
    except Exception as e:
        raise RuntimeError('Error while fetching file %s.\n'
                           'Dataset fetching aborted (%s)' % (url, e))
    finally:
        if local_file is not None:
            if not local_file.closed:
                local_file.close()