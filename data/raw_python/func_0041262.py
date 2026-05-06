def _json_to_subscription_post_response(response_body):
    """
    Returns a list of SubscriptionPostResponse objects
    """
    data = json.loads(response_body)
    response_list = []
    for response_data in data.get("responseList", []):
        response_list.append(SubscriptionPostResponse().from_json(
            data.get('uwNetID'), response_data))

    return response_list