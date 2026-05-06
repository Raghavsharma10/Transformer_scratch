def prettify_object(obj):
    """Makes a pretty string for an object for nice output"""

    try:
        return pprint.pformat(str(obj))
    except UnicodeDecodeError as e:
        raise
    except Exception as e:
        return "[could not display: <%s: %s>]" % (e.__class__.__name__, str(e))