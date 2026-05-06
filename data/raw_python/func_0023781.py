def get_resources_to_check(client_site_url, apikey):
    """Return a list of resource IDs to check for broken links.

    Calls the client site's API to get a list of resource IDs.

    :raises CouldNotGetResourceIDsError: if getting the resource IDs fails
        for any reason

    """
    url = client_site_url + u"deadoralive/get_resources_to_check"
    response = requests.get(url, headers=dict(Authorization=apikey))
    if not response.ok:
        raise CouldNotGetResourceIDsError(
            u"Couldn't get resource IDs to check: {code} {reason}".format(
                code=response.status_code, reason=response.reason))
    return response.json()