def get_url_for_id(client_site_url, apikey, resource_id):
    """Return the URL for the given resource ID.

    Contacts the client site's API to get the URL for the ID and returns it.

    :raises CouldNotGetURLError: if getting the URL fails for any reason

    """
    # TODO: Handle invalid responses from the client site.
    url = client_site_url + u"deadoralive/get_url_for_resource_id"
    params = {"resource_id": resource_id}
    response = requests.get(url, headers=dict(Authorization=apikey),
                            params=params)
    if not response.ok:
        raise CouldNotGetURLError(
            u"Couldn't get URL for resource {id}: {code} {reason}".format(
                id=resource_id, code=response.status_code,
                reason=response.reason))

    return response.json()