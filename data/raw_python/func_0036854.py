def make_unicode(s, encoding='utf-8', encoding_errors='strict'):
    """ Return the unicode version of an input. """
    if not isinstance(s, unicode):
        if not isinstance(s, basestring):
            return unicode(str(s), encoding, encoding_errors)
        return unicode(s, encoding, encoding_errors)
    return s