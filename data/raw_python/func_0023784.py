def upsert_result(client_site_url, apikey, resource_id, result):
    """Post the given link check result to the client site."""

    # TODO: Handle exceptions and unexpected results.
    url = client_site_url + u"deadoralive/upsert"
    params = result.copy()
    params["resource_id"] = resource_id
    requests.post(url, headers=dict(Authorization=apikey), params=params)