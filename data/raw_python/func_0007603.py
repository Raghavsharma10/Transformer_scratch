def upload(directory, metadata_csv, master_token=None, member=None,
           access_token=None, safe=False, sync=False, max_size='128m',
           mode='default', verbose=False, debug=False):
    """
    Upload files for the project to Open Humans member accounts.

    If using a master access token and not specifying member ID:

    (1) Files should be organized in subdirectories according to project
    member ID, e.g.:

        main_directory/01234567/data.json
        main_directory/12345678/data.json
        main_directory/23456789/data.json

    (2) The metadata CSV should have the following format:

        1st column: Project member ID
        2nd column: filenames
        3rd & additional columns: Metadata fields (see below)

    If uploading for a specific member:
    (1) The local directory should not contain subdirectories.
    (2) The metadata CSV should have the following format:
    1st column: filenames
    2nd & additional columns: Metadata fields (see below)

    The default behavior is to overwrite files with matching filenames on
    Open Humans, but not otherwise delete files. (Use --safe or --sync to
    change this behavior.)

    If included, the following metadata columns should be correctly formatted:
    'tags': should be comma-separated strings
    'md5': should match the file's md5 hexdigest
    'creation_date', 'start_date', 'end_date': ISO 8601 dates or datetimes

    Other metedata fields (e.g. 'description') can be arbitrary strings.
    Either specify sync as True or safe as True but not both.

    :param directory: This field is the target directory from which data will
        be uploaded.
    :param metadata_csv: This field is the filepath of the metadata csv file.
    :param master_token: This field is the master access token for the project.
        It's default value is None.
    :param member: This field is specific member whose project data is
        downloaded. It's default value is None.
    :param access_token: This field is the user specific access token. It's
        default value is None.
    :param safe: This boolean field will overwrite matching filename. It's
        default value is False.
    :param sync: This boolean field will delete files on Open Humans that are
        not in the local directory. It's default value is False.
    :param max_size: This field is the maximum file size. It's default value is
        None.
    :param mode: This field takes three value default, sync, safe. It's default
        value is 'default'.
    :param verbose: This boolean field is the logging level. It's default value
        is False.
    :param debug: This boolean field is the logging level. It's default value
        is False.
    """
    if safe and sync:
        raise UsageError('Safe (--safe) and sync (--sync) modes are mutually '
                         'incompatible!')
    if not (master_token or access_token) or (master_token and access_token):
        raise UsageError('Please specify either a master access token (-T), '
                         'or an OAuth2 user access token (-t).')

    set_log_level(debug, verbose)

    if sync:
        mode = 'sync'
    elif safe:
        mode = 'safe'

    metadata = load_metadata_csv(metadata_csv)

    subdirs = [i for i in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, i))]
    if subdirs:
        if not all([re.match(r'^[0-9]{8}$', d) for d in subdirs]):
            raise UsageError(
                "Subdirs expected to match project member ID format!")
        if (master_token and member) or not master_token:
            raise UsageError(
                "Subdirs shouldn't exist if uploading for specific member!")
        project = OHProject(master_access_token=master_token)
        for member_id in subdirs:
            subdir_path = os.path.join(directory, member_id)
            project.upload_member_from_dir(
                member_data=project.project_data[member_id],
                target_member_dir=subdir_path,
                metadata=metadata[member_id],
                mode=mode,
                access_token=project.master_access_token,
            )
    else:
        if master_token and not (master_token and member):
            raise UsageError('No member specified!')
        if master_token:
            project = OHProject(master_access_token=master_token)
            project.upload_member_from_dir(
                member_data=project.project_data[member],
                target_member_dir=directory,
                metadata=metadata,
                mode=mode,
                access_token=project.master_access_token,
            )
        else:
            member_data = exchange_oauth2_member(access_token)
            OHProject.upload_member_from_dir(
                member_data=member_data,
                target_member_dir=directory,
                metadata=metadata,
                mode=mode,
                access_token=access_token,
            )