def uriref_startswith_iriref(v1: URIRef, v2: Union[str, ShExJ.IRIREF]) -> bool:
    """ Determine whether a :py:class:`rdflib.URIRef` value starts with the text of a :py:class:`ShExJ.IRIREF` value """
    return str(v1).startswith(str(v2))