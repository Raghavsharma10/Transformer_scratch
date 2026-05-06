def write_zip(file_path, data, properties):
    """
        Write custom zip file to the file path

        :param file_path: the file to which to write the zip file
        :param data: the data to write to the file; may be None
        :param properties: the properties to write to the file; may be None

        The properties param must not change during this method. Callers should
        take care to ensure this does not happen.

        See write_zip_fp.
    """
    with open(file_path, "w+b") as fp:
        write_zip_fp(fp, data, properties)