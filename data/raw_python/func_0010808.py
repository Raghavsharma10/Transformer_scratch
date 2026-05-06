def smart_url(url, obj=None):
    """
    URLs that start with @ are reversed, using the passed in arguments.

    Otherwise a straight % substitution is applied.
    """
    if url.find("@") >= 0:
        (args, value) = url.split('@')

        if args:
            val = getattr(obj, args, None)
            return reverse(value, args=[val])
        else:
            return reverse(value)
    else:
        if obj is None:
            return url
        else:
            return url % obj.id