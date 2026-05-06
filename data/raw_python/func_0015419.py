def nodeInLanguageStem(_: Context, n: Node, s: ShExJ.LanguageStem) -> bool:
    """ http://shex.io/shex-semantics/#values

        **nodeIn**: asserts that an RDF node n is equal to an RDF term s or is in a set defined by a
        :py:class:`ShExJ.IriStem`, :py:class:`LiteralStem` or :py:class:`LanguageStem`.

        The expression `nodeInLanguageStem(n, s)` is satisfied iff:
         #) `s` is a :py:class:`ShExJ.WildCard` or
         #) `n` is a language-tagged string and fn:starts-with(`n.language`, `s`)
    """
    return isinstance(s, ShExJ.Wildcard) or \
        (isinstance(n, Literal) and n.language is not None and str(n.language).startswith(str(s)))