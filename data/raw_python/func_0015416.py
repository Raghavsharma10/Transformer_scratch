def _nodeSatisfiesValue(cntxt: Context, n: Node, vsv: ShExJ.valueSetValue) -> bool:
    """
    A term matches a valueSetValue if:
        * vsv is an objectValue and n = vsv.
        * vsv is a Language with langTag lt and n is a language-tagged string with a language tag l and l = lt.
        * vsv is a IriStem, LiteralStem or LanguageStem with stem st and nodeIn(n, st).
        * vsv is a IriStemRange, LiteralStemRange or LanguageStemRange with stem st and exclusions excls and
          nodeIn(n, st) and there is no x in excls such that nodeIn(n, excl).
        * vsv is a Wildcard with exclusions excls and there is no x in excls such that nodeIn(n, excl).

    Note that ObjectLiteral is *not* typed in ShExJ.jsg, so we identify it by a lack of a 'type' variable

    .. note:: Mismatch with spec
        This won't work correctly if the stem value is passed in to nodeIn, as there will be no way to know whether
        we're matching an IRI or other type

    ... note:: Language issue
        The stem range spec shouldn't have the first element in the exclusions

    """
    vsv = map_object_literal(vsv)
    if isinstance_(vsv, ShExJ.objectValue):
        return objectValueMatches(n, vsv)

    if isinstance(vsv, ShExJ.Language):
        if vsv.languageTag is not None and isinstance(n, Literal) and n.language is not None:
            return n.language == vsv.languageTag
        else:
            return False

    if isinstance(vsv, ShExJ.IriStem):
        return nodeInIriStem(cntxt, n, vsv.stem)

    if isinstance(vsv, ShExJ.IriStemRange):
        exclusions = vsv.exclusions if vsv.exclusions is not None else []
        return nodeInIriStem(cntxt, n, vsv.stem) and not any(
            (uriref_matches_iriref(n, excl) if isinstance(excl, ShExJ.IRIREF) else
             uriref_startswith_iriref(n, excl.stem)) for excl in exclusions)

    if isinstance(vsv, ShExJ.LiteralStem):
        return nodeInLiteralStem(cntxt, n, vsv.stem)

    if isinstance(vsv, ShExJ.LiteralStemRange):
        exclusions = vsv.exclusions if vsv.exclusions is not None else []
        return nodeInLiteralStem(cntxt, n, vsv.stem) and not any(str(n) == excl for excl in exclusions)

    if isinstance(vsv, ShExJ.LanguageStem):
        return nodeInLanguageStem(cntxt, n, vsv.stem)

    if isinstance(vsv, ShExJ.LanguageStemRange):
        exclusions = vsv.exclusions if vsv.exclusions is not None else []
        return nodeInLanguageStem(cntxt, n, vsv.stem) and not any(str(n) == str(excl) for excl in exclusions)

    return False