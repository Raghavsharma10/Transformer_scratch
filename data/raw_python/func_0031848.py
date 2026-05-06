def escapejson_filter(value):
    """
    Escape `value` to prevent </script> and unicode whitespace attacks. If
    `value` is not a string, JSON-encode it first.
    """
    if isinstance(value, six.string_types):
        string = value
    else:
        string = json.dumps(value, cls=DjangoJSONEncoder)
    return mark_safe(escapejson(string))