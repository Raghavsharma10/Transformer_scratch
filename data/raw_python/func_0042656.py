def _to_encoded_string(o):
    """
    Build an encoded string suitable for use as a URL component. This includes double-escaping the string to
    avoid issues with escaped backslash characters being automatically converted by WSGI or, in some cases
    such as default Apache servers, blocked entirely.

    :param o: an object of any kind, if it has an as_dict() method this will be used, otherwise uses __dict__
    :return: an encoded string suitable for use as a URL component
    :internal:
    """
    _dict = o.__dict__
    if o.as_dict:
        _dict = o.as_dict()
    return urllib.quote_plus(urllib.quote_plus(json.dumps(obj=_dict, separators=(',', ':'))))