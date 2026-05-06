def get_netid_subscriptions(netid, subscription_codes):
    """
    Returns a list of uwnetid.subscription objects
    corresponding to the netid and subscription code or list provided
    """
    url = _netid_subscription_url(netid, subscription_codes)
    response = get_resource(url)
    return _json_to_subscriptions(response)