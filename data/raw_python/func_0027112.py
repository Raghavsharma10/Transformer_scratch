def safe_str(source, max_length=0):
    """Wrapper for str() that catches exceptions."""
    try:
        return ellipsis(str(source), max_length)
    except Exception as e:
        return ellipsis("<n/a: str(...) raised %s>" % e, max_length)