def create_or_append_to_zip(file_handle, zip_path, arc_name=None):
    """
    Append file_handle to given zip_path with name arc_name if given, else file_handle. zip_path will be created.
    :param file_handle: path to file or file-like object
    :param zip_path: path to zip archive
    :param arc_name: optional filename in archive
    """
    with zipfile.ZipFile(zip_path, 'a') as my_zip:
        if arc_name:
            my_zip.write(file_handle, arc_name)
        else:
            my_zip.write(file_handle)