def to_camel(s):
    """
    :param string s: under_scored string to be CamelCased
    :return: CamelCase version of input
    :rtype: str
    """
    # r'(?!^)_([a-zA-Z]) original regex wasn't process first groups
    return re.sub(r'_([a-zA-Z])', lambda m: m.group(1).upper(), '_' + s)