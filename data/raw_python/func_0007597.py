def download(directory, master_token=None, member=None, access_token=None,
             source=None, project_data=False, max_size='128m', verbose=False,
             debug=False, memberlist=None, excludelist=None,
             id_filename=False):
    """
    Download data from project members to the target directory.

    Unless this is a member-specific download, directories will be
    created for each project member ID. Also, unless a source is specified,
    all shared sources are downloaded and data is sorted into subdirectories
    according to source.

    Projects can optionally return data to Open Humans member accounts.
    If project_data is True (or the "--project-data" flag is used), this data
    (the project's own data files, instead of data from other sources) will be
    downloaded for each member.

    :param directory: This field is the target directory to download data.
    :param master_token: This field is the master access token for the project.
        It's default value is None.
    :param member: This field is specific member whose project data is
        downloaded. It's default value is None.
    :param access_token: This field is the user specific access token. It's
        default value is None.
    :param source: This field is the data source. It's default value is None.
    :param project_data: This field is data related to particular project. It's
        default value is False.
    :param max_size: This field is the maximum file size. It's default value is
        128m.
    :param verbose: This boolean field is the logging level. It's default value
        is False.
    :param debug: This boolean field is the logging level. It's default value
        is False.
    :param memberlist: This field is list of members whose data will be
        downloaded. It's default value is None.
    :param excludelist: This field is list of members whose data will be
        skipped. It's default value is None.
    """
    set_log_level(debug, verbose)

    if (memberlist or excludelist) and (member or access_token):
        raise UsageError('Please do not provide a memberlist or excludelist '
                         'when retrieving data for a single member.')
    memberlist = read_id_list(memberlist)
    excludelist = read_id_list(excludelist)
    if not (master_token or access_token) or (master_token and access_token):
        raise UsageError('Please specify either a master access token (-T), '
                         'or an OAuth2 user access token (-t).')
    if (source and project_data):
        raise UsageError("It doesn't make sense to use both 'source' and"
                         "'project-data' options!")

    if master_token:
        project = OHProject(master_access_token=master_token)
        if member:
            if project_data:
                project.download_member_project_data(
                    member_data=project.project_data[member],
                    target_member_dir=directory,
                    max_size=max_size,
                    id_filename=id_filename)
            else:
                project.download_member_shared(
                    member_data=project.project_data[member],
                    target_member_dir=directory,
                    source=source,
                    max_size=max_size,
                    id_filename=id_filename)
        else:
            project.download_all(target_dir=directory,
                                 source=source,
                                 max_size=max_size,
                                 memberlist=memberlist,
                                 excludelist=excludelist,
                                 project_data=project_data,
                                 id_filename=id_filename)
    else:
        member_data = exchange_oauth2_member(access_token, all_files=True)
        if project_data:
            OHProject.download_member_project_data(member_data=member_data,
                                                   target_member_dir=directory,
                                                   max_size=max_size,
                                                   id_filename=id_filename)
        else:
            OHProject.download_member_shared(member_data=member_data,
                                             target_member_dir=directory,
                                             source=source,
                                             max_size=max_size,
                                             id_filename=id_filename)