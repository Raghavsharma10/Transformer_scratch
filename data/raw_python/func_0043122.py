def parse_qs(string):
    """Intelligently parse the query string"""
    result = {}

    for item in split_qs(string):
        # Split the query string by unquotes ampersants ('&')
        try:
            # Split the item by unquotes equal signs
            key, value = split_qs(item, delimiter='=')
        except ValueError:
            # Single value without equals sign
            result[item] = ''
        else:
            result[key] = value

    return result