def nodeInIriStem(_: Context, n: Node, s: ShExJ.IriStem) -> bool:
    """
       **nodeIn**: asserts that an RDF node n is equal to an RDF term s or is in a set defined by a
       :py:class:`ShExJ.IriStem`, :py:class:`LiteralStem` or :py:class:`LanguageStem`.

       The expression `nodeInIriStem(n, s)` is satisfied iff:
        #) `s` is a :py:class:`ShExJ.WildCard` or
        #) `n` is an :py:class:`rdflib.URIRef` and fn:starts-with(`n`, `s`)
    """
    return isinstance(s, ShExJ.Wildcard) or \
        (isinstance(n, URIRef) and uriref_startswith_iriref(n, str(s)))