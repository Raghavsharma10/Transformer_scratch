def normalize_uriparm(p: URIPARM) -> List[URIRef]:
    """ Return an optional list of URIRefs for p"""
    return normalize_urilist(p) if isinstance(p, List) else \
        normalize_urilist([p]) if isinstance(p, (str, URIRef)) else p