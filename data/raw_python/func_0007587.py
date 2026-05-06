def review_metadata_csv_single_user(filedir, metadata, csv_in, n_headers):
    """
    Check validity of metadata for single user.

    :param filedir: This field is the filepath of the directory whose csv
        has to be made.
    :param metadata: This field is the metadata generated from the
        load_metadata_csv function.
    :param csv_in: This field returns a reader object which iterates over the
        csv.
    :param n_headers: This field is the number of headers in the csv.
    """
    try:
        if not validate_metadata(filedir, metadata):
            return False
        for filename, file_metadata in metadata.items():
            is_single_file_metadata_valid(file_metadata, None, filename)
    except ValueError as e:
        print_error(e)
        return False
    return True