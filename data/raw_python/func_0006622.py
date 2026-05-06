def _uax44lm2transform(s):
    """
    Helper function for taking a string (i.e. a Unicode character name) and transforming it via UAX44-LM2 loose matching
    rule.  For more information, see <https://www.unicode.org/reports/tr44/#UAX44-LM2>.

    The rule is defined as follows:

    "UAX44-LM2. Ignore case, whitespace, underscore ('_'), and all medial hyphens except the hyphen in
    U+1180 HANGUL JUNGSEONG O-E."

    Therefore, correctly implementing the rule involves performing the following three operations, in order:

    1. remove all medial hyphens (except the medial hyphen in the name for U+1180)
    2. remove all whitespace and underscore characters
    3. apply toLowercase() to both strings

    A "medial hyphen" is defined as follows (quoted from the above referenced web page):

    "In this rule 'medial hyphen' is to be construed as a hyphen occurring immediately between two letters in the
    normative Unicode character name, as published in the Unicode names list, and not to any hyphen that may transiently
    occur medially as a result of removing whitespace before removing hyphens in a particular implementation of
    matching. Thus the hyphen in the name U+10089 LINEAR B IDEOGRAM B107M HE-GOAT is medial, and should be ignored in
    loose matching, but the hyphen in the name U+0F39 TIBETAN MARK TSA -PHRU is not medial, and should not be ignored in
    loose matching."


    :param s: String to transform
    :return: String transformed per UAX44-LM2 loose matching rule.
    """
    result = s

    # For the regex, we are using lookaround assertions to verify that there is a word character immediately before (the
    # lookbehind assertion (?<=\w)) and immediately after (the lookahead assertion (?=\w)) the hyphen, per the "medial
    # hyphen" definition that it is a hyphen occurring immediately between two letters.
    medialhyphen = re.compile(r"(?<=\w)-(?=\w)")
    whitespaceunderscore = re.compile(r"[\s_]", re.UNICODE)

    # Ok to hard code, this name should never change: https://www.unicode.org/policies/stability_policy.html#Name
    if result != "HANGUL JUNGSEONG O-E":
        result = medialhyphen.sub("", result)
    result = whitespaceunderscore.sub("", result)
    return result.lower()