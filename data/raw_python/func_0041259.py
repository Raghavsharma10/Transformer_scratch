def update_subscription(netid, action, subscription_code, data_field=None):
    """
    Post a subscription action for the given netid and subscription_code
    """
    url = '{0}/subscription.json'.format(url_version())
    action_list = []

    if isinstance(subscription_code, list):
        for code in subscription_code:
            action_list.append(_set_action(
                netid, action, code, data_field))
    else:
        action_list.append(_set_action(
            netid, action, subscription_code, data_field))

    body = {'actionList': action_list}
    response = post_resource(url, json.dumps(body))
    return _json_to_subscription_post_response(response)