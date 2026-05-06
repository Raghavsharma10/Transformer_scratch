def _decode_POST_value(request, field_name, default=None):
    """Helper to decode a request field into unicode based on charsets encoding.

    Args:
        request: the HttpRequest object.
        field_name: the field expected in the request.POST

    Kwargs:
        default: if passed in then field is optional and default is used if not
            found; if None, then assume field exists, which will raise an error
            if it does not.

    Returns: the contents of the string encoded using the related charset from
        the requests.POST['charsets'] dictionary (or 'utf-8' if none specified).
    """
    if default is None:
        value = request.POST[field_name]
    else:
        value = request.POST.get(field_name, default)

    # it's inefficient to load this each time it gets called, but we're
    # not anticipating incoming email being a performance bottleneck right now!
    charsets = json.loads(request.POST.get('charsets', "{}"))
    charset = charsets.get(field_name, 'utf-8')

    if charset.lower() != 'utf-8':
        logger.debug("Incoming email field '%s' has %s encoding.", field_name, charset)

    return smart_text(value, encoding=charset)