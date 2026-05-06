def validate(document, spec):
    """Validate that a document meets a specification.  Returns True if
    validation was successful, but otherwise raises a ValueError."""
    if not spec:
        return True
    missing = []
    for key, field in spec.iteritems():
        if field.required and key not in document:
            missing.append(key)

    failed = []
    for key, field in spec.iteritems():
        if key in document:
            try: document[key] = field.validate(document[key])
            except ValueError: failed.append(key)

    if missing or failed:
        if missing and not failed:
            raise ValueError("Required fields missing: %s" % (missing))
        if failed and not missing:
            raise ValueError("Keys did not match spec: %s" % (failed))
        raise ValueError("Missing fields: %s, Invalid fields: %s" % (missing, failed))
    # just a token of my kindness, a return for you
    return True