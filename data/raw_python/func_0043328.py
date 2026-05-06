def deserialize_uri(value):
    """
    Deserialize a representation of a BNode or URIRef.
    """
    if isinstance(value, BNode):
        return value
    if isinstance(value, URIRef):
        return value
    if not value:
        return None
    if not isinstance(value, basestring):
        raise ValueError("Cannot create URI from {0} of type {1}".format(value, value.__class__))
    if value.startswith("_:"):
        return BNode(value[2:])
    return URIRef(value)