def objectValueMatches(n: Node, vsv: ShExJ.objectValue) -> bool:
    """ http://shex.io/shex-semantics/#values

    Implements "n = vsv" where vsv is an objectValue and n is a Node

    Note that IRIREF is a string pattern, so the matching type is str
    """
    return \
        (isinstance(vsv, IRIREF) and isinstance(n, URIRef) and uriref_matches_iriref(n, vsv)) or \
        (isinstance(vsv, ShExJ.ObjectLiteral) and isinstance(n, Literal) and literal_matches_objectliteral(n, vsv))