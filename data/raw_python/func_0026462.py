def get_project_build(account_project):
    """Get the details of the latest Appveyor build."""
    url = make_url("/projects/{account_project}", account_project=account_project)
    response = requests.get(url, headers=make_auth_headers())
    return response.json()