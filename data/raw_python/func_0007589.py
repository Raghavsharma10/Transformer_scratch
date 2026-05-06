def review_metadata_csv_multi_user(filedir, metadata, csv_in, n_headers):
    """
    Check validity of metadata for multi user.

    :param filedir: This field is the filepath of the directory whose csv
        has to be made.
    :param metadata: This field is the metadata generated from the
        load_metadata_csv function.
    :param csv_in: This field returns a reader object which iterates over the
        csv.
    :param n_headers: This field is the number of headers in the csv.
    """
    try:
        if not validate_subfolders(filedir, metadata):
            return False
        for project_member_id, member_metadata in metadata.items():
            if not validate_metadata(os.path.join
                                     (filedir, project_member_id),
                                     member_metadata):
                return False
            for filename, file_metadata in member_metadata.items():
                is_single_file_metadata_valid(file_metadata, project_member_id,
                                              filename)

    except ValueError as e:
        print_error(e)
        return False
    return True