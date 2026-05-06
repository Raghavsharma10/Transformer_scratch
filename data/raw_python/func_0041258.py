def modify_subscription_status(netid, subscription_code, status):
    """
    Post a subscription 'modify' action for the given netid
    and subscription_code
    """
    url = _netid_subscription_url(netid, subscription_code)
    body = {
        'action': 'modify',
        'value': str(status)
    }

    response = post_resource(url, json.dumps(body))
    return _json_to_subscriptions(response)