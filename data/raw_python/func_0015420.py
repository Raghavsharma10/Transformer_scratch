def nodeInBnodeStem(_cntxt: Context, _n: Node, _s: Union[str, ShExJ.Wildcard]) -> bool:
    """ http://shex.io/shex-semantics/#values

        **nodeIn**: asserts that an RDF node n is equal to an RDF term s or is in a set defined by a
        :py:class:`ShExJ.IriStem`, :py:class:`LiteralStem` or :py:class:`LanguageStem`.

        The expression `nodeInBnodeStem(n, s)` is satisfied iff:
         #) `s` is a :py:class:`ShExJ.WildCard` or
         #) `n` is a language-tagged string and fn:starts-with(`n.language`, `s`)

    """
    # TODO: resolve issue #79 to figure out how to do this
    return False