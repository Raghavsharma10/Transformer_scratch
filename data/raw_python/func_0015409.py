def literal_matches_objectliteral(v1: Literal, v2: ShExJ.ObjectLiteral) -> bool:
    """ Compare :py:class:`rdflib.Literal` with :py:class:`ShExJ.objectLiteral` """
    v2_lit = Literal(str(v2.value), datatype=iriref_to_uriref(v2.type), lang=str(v2.language) if v2.language else None)
    return v1 == v2_lit