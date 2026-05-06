def serialize_uri(value):
    """
    Serialize a BNode or URIRef.
    """
    if isinstance(value, BNode):
        return value.n3()
    if isinstance(value, URIRef):
        return unicode(value)
    raise ValueError("Cannot get prepvalue for {0} of type {1}".format(value, value.__class__))