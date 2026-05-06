def queues_for_endpoint(event):
    """
    Return the list of queues to publish to for a given endpoint.

    :param event: Lambda event that triggered the handler
    :type event: dict
    :return: list of queues for endpoint
    :rtype: :std:term:`list`
    :raises: Exception
    """
    global endpoints  # endpoint config that's templated in by generator
    # get endpoint config
    try:
        ep_name = event['context']['resource-path'].lstrip('/')
        return endpoints[ep_name]['queues']
    except:
        raise Exception('Endpoint not in configuration: /%s' % ep_name)