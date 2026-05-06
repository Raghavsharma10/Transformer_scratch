def delete_file(access_token, project_member_id=None, base_url=OH_BASE_URL,
                file_basename=None, file_id=None, all_files=False):
    """
    Delete project member files by file_basename, file_id, or all_files. To
        learn more about Open Humans OAuth2 projects, go to:
        https://www.openhumans.org/direct-sharing/oauth2-features/.

    :param access_token: This field is user specific access_token.
    :param project_member_id: This field is the project member id of user. It's
        default value is None.
    :param base_url: It is this URL `https://www.openhumans.org`.
    :param file_basename: This field is the name of the file to delete for the
        particular user for the particular project.
    :param file_id: This field is the id of the file to delete for the
        particular user for the particular project.
    :param all_files: This is a boolean field to delete all files for the
        particular user for the particular project.
    """
    url = urlparse.urljoin(
        base_url, '/api/direct-sharing/project/files/delete/?{}'.format(
            urlparse.urlencode({'access_token': access_token})))
    if not(project_member_id):
        response = exchange_oauth2_member(access_token, base_url=base_url)
        project_member_id = response['project_member_id']
    data = {'project_member_id': project_member_id}
    if file_basename and not (file_id or all_files):
        data['file_basename'] = file_basename
    elif file_id and not (file_basename or all_files):
        data['file_id'] = file_id
    elif all_files and not (file_id or file_basename):
        data['all_files'] = True
    else:
        raise ValueError(
            "One (and only one) of the following must be specified: "
            "file_basename, file_id, or all_files is set to True.")
    response = requests.post(url, data=data)
    handle_error(response, 200)
    return response