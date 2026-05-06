def upload_aws(target_filepath, metadata, access_token, base_url=OH_BASE_URL,
               remote_file_info=None, project_member_id=None,
               max_bytes=MAX_FILE_DEFAULT):
    """
    Upload a file from a local filepath using the "direct upload" API.
    Equivalent to upload_file. To learn more about this API endpoint see:
    * https://www.openhumans.org/direct-sharing/on-site-data-upload/
    * https://www.openhumans.org/direct-sharing/oauth2-data-upload/

    :param target_filepath: This field is the filepath of the file to be
        uploaded
    :param metadata: This field is the metadata associated with the file.
        Description and tags are compulsory fields of metadata.
    :param access_token: This is user specific access token/master token.
    :param base_url: It is this URL `https://www.openhumans.org`.
    :param remote_file_info: This field is for for checking if a file with
        matching name and file size already exists. Its default value is none.
    :param project_member_id: This field is the list of project member id of
        all members of a project. Its default value is None.
    :param max_bytes: This field is the maximum file size a user can upload.
        It's default value is 128m.
    """
    return upload_file(target_filepath, metadata, access_token, base_url,
                       remote_file_info, project_member_id, max_bytes)