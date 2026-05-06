def rewrite_zip(file_path, properties):
    """
        Rewrite the json properties in the zip file

        :param file_path: the file path to the zip file
        :param properties: the updated properties to write to the zip file

        This method will attempt to keep the data file within the zip
        file intact without rewriting it. However, if the data file is not the
        first item in the zip file, this method will rewrite it.

        The properties param must not change during this method. Callers should
        take care to ensure this does not happen.
    """
    with open(file_path, "r+b") as fp:
        local_files, dir_files, eocd = parse_zip(fp)
        # check to make sure directory has two files, named data.npy and metadata.json, and that data.npy is first
        # TODO: check compression, etc.
        if len(dir_files) == 2 and b"data.npy" in dir_files and b"metadata.json" in dir_files and dir_files[b"data.npy"][1] == 0:
            fp.seek(dir_files[b"metadata.json"][1])
            dir_data_list = list()
            local_file_pos = dir_files[b"data.npy"][1]
            local_file = local_files[local_file_pos]
            dir_data_list.append((local_file_pos, b"data.npy", local_file[2], local_file[3]))
            write_zip_fp(fp, None, properties, dir_data_list)
        else:
            data = None
            if b"data.npy" in dir_files:
                fp.seek(local_files[dir_files[b"data.npy"][1]][1])
                data = numpy.load(fp)
            fp.seek(0)
            write_zip_fp(fp, data, properties)