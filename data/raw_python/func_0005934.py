def output_capturing():
    """Temporarily captures/redirects stdout."""
    out = sys.stdout

    sys.stdout = StringIO()

    try:
        yield

    finally:
        sys.stdout = out