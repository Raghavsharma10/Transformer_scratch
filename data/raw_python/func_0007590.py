def review_metadata_csv(filedir, input_filepath):
    """
    Check validity of metadata fields.

    :param filedir: This field is the filepath of the directory whose csv
        has to be made.
    :param outputfilepath: This field is the file path of the output csv.
    :param max_bytes: This field is the maximum file size to consider. Its
        default value is 128m.
    """
    try:
        metadata = load_metadata_csv(input_filepath)
    except ValueError as e:
        print_error(e)
        return False

    with open(input_filepath) as f:
        csv_in = csv.reader(f)
        header = next(csv_in)
        n_headers = len(header)
        if header[0] == 'filename':
            res = review_metadata_csv_single_user(filedir, metadata,
                                                  csv_in, n_headers)
            return res
        if header[0] == 'project_member_id':
            res = review_metadata_csv_multi_user(filedir, metadata,
                                                 csv_in, n_headers)
            return res