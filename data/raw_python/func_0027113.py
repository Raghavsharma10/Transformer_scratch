def safe_repr(source, max_length=0):
    """Wrapper for repr() that catches exceptions."""
    try:
        return ellipsis(repr(source), max_length)
    except Exception as e:
        return ellipsis("<n/a: repr(...) raised %s>" % e, max_length)