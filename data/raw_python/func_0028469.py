def _get_results_from_api(identifiers, endpoints, api_key, api_secret):
    """Use the HouseCanary API Python Client to access the API"""

    if api_key is not None and api_secret is not None:
        client = housecanary.ApiClient(api_key, api_secret)
    else:
        client = housecanary.ApiClient()

    wrapper = getattr(client, endpoints[0].split('/')[0])

    if len(endpoints) > 1:
        # use component_mget to request multiple endpoints in one call
        return wrapper.component_mget(identifiers, endpoints)
    else:
        return wrapper.fetch_identifier_component(endpoints[0], identifiers)