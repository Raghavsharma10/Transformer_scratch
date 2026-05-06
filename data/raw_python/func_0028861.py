def ppj(json_data):
    """ppj

    :param json_data: dictionary to print
    """
    return str(json.dumps(
                json_data,
                sort_keys=True,
                indent=4,
                separators=(',', ': ')))