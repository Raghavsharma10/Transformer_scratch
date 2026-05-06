def check_result(data, key=''):
    """Check the result of an API response.

    Ideally, this should be done by checking that the value of the ``resultCode``
    attribute is 0, but there are endpoints that simply do not follow this rule.

    Args:
        data (dict): Response obtained from the API endpoint.
        key (string): Key to check for existence in the dict.

    Returns:
        bool: True if result was correct, False otherwise.
    """
    if not isinstance(data, dict):
        return False

    if key:
        if key in data:
            return True

        return False

    if 'resultCode' in data.keys():
        # OpenBus
        return True if data.get('resultCode', -1) == 0 else False

    elif 'code' in data.keys():
        # Parking
        return True if data.get('code', -1) == 0 else False

    return False