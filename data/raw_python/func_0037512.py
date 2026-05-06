def text_cleanup(data, key, last_type):
    """ I strip extra whitespace off multi-line strings if they are ready to be stripped!"""
    if key in data and last_type == STRING_TYPE:
        data[key] = data[key].strip()
    return data