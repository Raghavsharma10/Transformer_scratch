def casefold(s, fullcasefold=True, useturkicmapping=False):
    """
    Function for performing case folding.  This function will take the input
    string s and return a copy of the string suitable for caseless comparisons.
    The input string must be of type 'unicode', otherwise a TypeError will be
    raised.

    For more information on case folding, see section 3.13 of the Unicode Standard.
    See also the following FAQ on the Unicode website:

    https://unicode.org/faq/casemap_charprop.htm

    By default, full case folding (where the string length may change) is done.
    It is possible to use simple case folding (single character mappings only)
    by setting the boolean parameter fullcasefold=False.

    By default, case folding does not handle the Turkic case of dotted vs dotless 'i'.
    To perform case folding using the special Turkic mappings, pass the boolean
    parameter useturkicmapping=True.  For more info on the dotted vs dotless 'i', see
    the following web pages:

    https://en.wikipedia.org/wiki/Dotted_and_dotless_I
    http://www.i18nguy.com/unicode/turkish-i18n.html#problem

    :param s: String to transform
    :param fullcasefold: Boolean indicating if a full case fold (default is True) should be done.  If False, a simple
                         case fold will be performed.
    :param useturkicmapping: Boolean indicating if the special turkic mapping (default is False) for the dotted and
                             dotless 'i' should be used.
    :return: Copy of string that has been transformed for caseless comparison.
    """
    if not isinstance(s, six.text_type):
        raise TypeError(u"String to casefold must be of type 'unicode'!")
    lookup_order = "CF"
    if not fullcasefold:
        lookup_order = "CS"
    if useturkicmapping:
        lookup_order = "T" + lookup_order
    return u"".join([casefold_map.lookup(c, lookup_order=lookup_order) for c in preservesurrogates(s)])