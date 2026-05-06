def normalize_uri(u: URI) -> URIRef:
    """ Return a URIRef for a str or URIRef """
    return u if isinstance(u, URIRef) else URIRef(str(u))