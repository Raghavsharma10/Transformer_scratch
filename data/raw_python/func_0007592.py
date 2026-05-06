def mk_metadata_csv(filedir, outputfilepath, max_bytes=MAX_FILE_DEFAULT):
    """
    Make metadata file for all files in a directory.

    :param filedir: This field is the filepath of the directory whose csv
        has to be made.
    :param outputfilepath: This field is the file path of the output csv.
    :param max_bytes: This field is the maximum file size to consider. Its
        default value is 128m.
    """
    with open(outputfilepath, 'w') as filestream:
        write_metadata_to_filestream(filedir, filestream, max_bytes)