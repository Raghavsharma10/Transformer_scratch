def unaccent(string, encoding="utf-8"):
    """not just unaccent, but full to-ascii transliteration"""
    string = to_unicode(string)
    if has_unidecode:
        return unidecode.unidecode(string)
    if PYTHON_VERSION < 3:
        if type(string) == str:
            string = unicode(string, encoding)
        nfkd_form = unicodedata.normalize('NFKD', string)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)]).encode("ascii", "ignore")
    else:
        return string