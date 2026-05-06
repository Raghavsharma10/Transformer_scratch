def uriref_matches_iriref(v1: URIRef, v2: Union[str, ShExJ.IRIREF]) -> bool:
    """ Compare :py:class:`rdflib.URIRef` value with :py:class:`ShExJ.IRIREF` value """
    return str(v1) == str(v2)