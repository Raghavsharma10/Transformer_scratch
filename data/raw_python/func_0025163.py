def _get_uploaded_versions_pypicloud(project_name, index_url, requests_verify=True):
    """ Query the pypi index at index_url using pypicloud api to find all versions """
    api_url = index_url
    for suffix in ('/pypi', '/pypi/', '/simple', '/simple/'):
        if api_url.endswith(suffix):
            api_url = api_url[:len(suffix) * -1] + '/api/package'
            break
    url = '/'.join((api_url, project_name))
    response = requests.get(url, verify=requests_verify)
    if response.status_code == 200:
        return [p['version'] for p in response.json()['packages']]
    return None