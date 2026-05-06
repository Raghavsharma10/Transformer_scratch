def _get_uploaded_versions_warehouse(project_name, index_url, requests_verify=True):
    """ Query the pypi index at index_url using warehouse api to find all of the "releases" """
    url = '/'.join((index_url, project_name, 'json'))
    response = requests.get(url, verify=requests_verify)
    if response.status_code == 200:
        return response.json()['releases'].keys()
    return None