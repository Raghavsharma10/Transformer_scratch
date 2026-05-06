def expand(cls, match, expand):
        """
        If use expand directly, the url-decoded context will be decoded again, which create a security
        issue. Hack expand to quote the text before expanding
        """
        return re._expand(match.re, cls._EncodedMatch(match), expand)