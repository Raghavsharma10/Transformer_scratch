def service_url_parse(url):
    """
    Function that parses from url the service and folder of services.
    """
    endpoint = get_sanitized_endpoint(url)
    url_split_list = url.split(endpoint + '/')
    if len(url_split_list) != 0:
        url_split_list = url_split_list[1].split('/')
    else:
        raise Exception('Wrong url parsed')

    # Remove unnecessary items from list of the split url.
    parsed_url = [s for s in url_split_list if '?' not in s if 'Server' not in s]

    return parsed_url