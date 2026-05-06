def nodeInLiteralStem(_: Context, n: Node, s: ShExJ.LiteralStem) -> bool:
    """ http://shex.io/shex-semantics/#values

        **nodeIn**: asserts that an RDF node n is equal to an RDF term s or is in a set defined by a
        :py:class:`ShExJ.IriStem`, :py:class:`LiteralStem` or :py:class:`LanguageStem`.

        The expression `nodeInLiteralStem(n, s)` is satisfied iff:
         #) `s` is a :py:class:`ShExJ.WildCard` or
         #) `n` is an :py:class:`rdflib.Literal` and fn:starts-with(`n`, `s`)
     """
    return isinstance(s, ShExJ.Wildcard) or \
        (isinstance(n, Literal) and str(n.value).startswith(str(s)))