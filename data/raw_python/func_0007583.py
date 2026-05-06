def load_metadata_csv_single_user(csv_in, header, tags_idx):
    """
    Return the metadata as requested for a single user.

    :param csv_in: This field is the csv file to return metadata from.
    :param header: This field contains the headers in the csv file
    :param tags_idx: This field contains the index of the tags in the csv
        file.
    """
    metadata = {}
    n_headers = len(header)
    for index, row in enumerate(csv_in, 2):
        if row[0] == "":
            raise ValueError('Error: In row number ' + str(index) + ':' +
                             ' "filename" must not be empty.')
        if row[0] == 'None' and [x == 'NA' for x in row[1:]]:
            break
        if len(row) != n_headers:
            raise ValueError('Error: In row number ' + str(index) + ':' +
                             ' Number of columns (' + str(len(row)) +
                             ') doesnt match Number of headings (' +
                             str(n_headers) + ')')
        metadata[row[0]] = {
            header[i]: row[i] for i in range(1, len(header)) if
            i != tags_idx
        }
        metadata[row[0]]['tags'] = [t.strip() for t in
                                    row[tags_idx].split(',') if
                                    t.strip()]
    return metadata