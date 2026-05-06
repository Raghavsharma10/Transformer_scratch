def require_json():
    """ Load the best available json library on demand.
    """
    # Fails when "json" is missing and "simplejson" is not installed either
    try:
        import json # pylint: disable=F0401
        return json
    except ImportError:
        try:
            import simplejson # pylint: disable=F0401
            return simplejson
        except ImportError as exc:
            raise ImportError("""Please 'pip install "simplejson>=2.1.6"' (%s)""" % (exc,))