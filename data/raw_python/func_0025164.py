def version_already_uploaded(project_name, version_str, index_url, requests_verify=True):
    """ Check to see if the version specified has already been uploaded to the configured index
    """
    all_versions = _get_uploaded_versions(project_name, index_url, requests_verify)
    return version_str in all_versions