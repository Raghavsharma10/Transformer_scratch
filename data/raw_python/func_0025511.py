def read_json(fp, local_files, dir_files, name_bytes):
    """
        Read json properties from the zip file

        :param fp: a file pointer
        :param local_files: the local files structure
        :param dir_files: the directory headers
        :param name: the name of the json file to read
        :return: the json properites as a dictionary, if found

        The file pointer will be at a location following the
        local file entry after this method.

        The local_files and dir_files should be passed from
        the results of parse_zip.
    """
    if name_bytes in dir_files:
        json_pos = local_files[dir_files[name_bytes][1]][1]
        json_len = local_files[dir_files[name_bytes][1]][2]
        fp.seek(json_pos)
        json_properties = fp.read(json_len)
        return json.loads(json_properties.decode("utf-8"))
    return None