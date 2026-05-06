def _prepare_params(dirty_params, prefix=None):
    """Prepares parameters to be sent to challonge.com.

    The `prefix` can be used to convert parameters with keys that
    look like ("name", "url", "tournament_type") into something like
    ("tournament[name]", "tournament[url]", "tournament[tournament_type]"),
    which is how challonge.com expects parameters describing specific
    objects.

    """
    params = {}
    for k, v in dirty_params.items():
        if hasattr(v, "isoformat"):
            v = v.isoformat()
        elif isinstance(v, bool):
            # challonge.com only accepts lowercase true/false
            v = str(v).lower()

        if prefix:
            params["%s[%s]" % (prefix, k)] = v
        else:
            params[k] = v

    return params