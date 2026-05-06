def read_data(fp, local_files, dir_files, name_bytes):
    """
        Read a numpy data array from the zip file

        :param fp: a file pointer
        :param local_files: the local files structure
        :param dir_files: the directory headers
        :param name: the name of the data file to read
        :return: the numpy data array, if found

        The file pointer will be at a location following the
        local file entry after this method.

        The local_files and dir_files should be passed from
        the results of parse_zip.
    """
    if name_bytes in dir_files:
        fp.seek(local_files[dir_files[name_bytes][1]][1])
        return numpy.load(fp)
    return None