def _json_to_subscriptions(response_body):
    """
    Returns a list of Subscription objects
    """
    data = json.loads(response_body)
    subscriptions = []
    for subscription_data in data.get("subscriptionList", []):
        subscriptions.append(Subscription().from_json(
            data.get('uwNetID'), subscription_data))

    return subscriptions