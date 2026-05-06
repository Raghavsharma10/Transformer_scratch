def make_default(spec):
    """Create an empty document that follows spec.  Any field with a default
    will take that value, required or not.  Required fields with no default
    will get a value of None.  If your default value does not match your
    type or otherwise customized Field class, this can create a spec that
    fails validation."""
    doc = {}
    for key, field in spec.iteritems():
        if field.default is not no_default:
            doc[key] = field.default
    return doc