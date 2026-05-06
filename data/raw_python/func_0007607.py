def delete_cli(access_token, project_member_id, base_url=OH_BASE_URL,
               file_basename=None, file_id=None, all_files=False):
    """
    Command line function for deleting files.
    For more information visit
    :func:`delete_file<ohapi.api.delete_file>`.
    """
    response = delete_file(access_token, project_member_id,
                           base_url, file_basename, file_id, all_files)
    if (response.status_code == 200):
        print("File deleted successfully")
    else:
        print("Bad response while deleting file.")