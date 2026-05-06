def is_single_file_metadata_valid(file_metadata, project_member_id, filename):
    """
    Check if metadata fields like project member id, description, tags, md5 and
    creation date are valid for a single file.

    :param file_metadata: This field is metadata of file.
    :param project_member_id: This field is the project member id corresponding
        to the file metadata provided.
    :param filename: This field is the filename corresponding to the file
        metadata provided.
    """
    if project_member_id is not None:
        if not project_member_id.isdigit() or len(project_member_id) != 8:
            raise ValueError(
                'Error: for project member id: ', project_member_id,
                ' and filename: ', filename,
                ' project member id must be of 8 digits from 0 to 9')
    if 'description' not in file_metadata:
        raise ValueError(
            'Error: for project member id: ', project_member_id,
            ' and filename: ', filename,
            ' "description" is a required field of the metadata')

    if not isinstance(file_metadata['description'], str):
        raise ValueError(
            'Error: for project member id: ', project_member_id,
            ' and filename: ', filename,
            ' "description" must be a string')

    if 'tags' not in file_metadata:
        raise ValueError(
            'Error: for project member id: ', project_member_id,
            ' and filename: ', filename,
            ' "tags" is a required field of the metadata')

    if not isinstance(file_metadata['tags'], list):
        raise ValueError(
            'Error: for project member id: ', project_member_id,
            ' and filename: ', filename,
            ' "tags" must be an array of strings')

    if 'creation_date' in file_metadata:
        if not validate_date(file_metadata['creation_date'], project_member_id,
                             filename):
            raise ValueError(
                'Error: for project member id: ', project_member_id,
                ' and filename: ', filename,
                ' Dates must be in ISO 8601 format')

    if 'md5' in file_metadata:
        if not re.match(r'[a-f0-9]{32}$', file_metadata['md5'],
                        flags=re.IGNORECASE):
            raise ValueError(
                'Error: for project member id: ', project_member_id,
                ' and filename: ', filename,
                ' Invalid MD5 specified')

    return True