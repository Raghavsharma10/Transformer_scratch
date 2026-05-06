def create_regex_patterns(symbols):
    u"""create regex patterns for text, google, docomo, kddi and softbank via `symbols`
    
    create regex patterns for finding emoji character from text. the pattern character use
    `unicode` formatted character so you have to decode text which is not decoded.
    """
    pattern_unicode = []
    pattern_google = []
    pattern_docomo = []
    pattern_kddi = []
    pattern_softbank = []
    for x in symbols:
        if x.unicode.code: pattern_unicode.append(re.escape(unicode(x.unicode)))
        if x.google.code: pattern_google.append(re.escape(unicode(x.google)))
        if x.docomo.code: pattern_docomo.append(re.escape(unicode(x.docomo)))
        if x.kddi.code: pattern_kddi.append(re.escape(unicode(x.kddi)))
        if x.softbank.code: pattern_softbank.append(re.escape(unicode(x.softbank)))
#    pattern_unicode = re.compile(u"[%s]" % u''.join(pattern_unicode))
#    pattern_google = re.compile(u"[%s]" % u''.join(pattern_google))
#    pattern_docomo = re.compile(u"[%s]" % u''.join(pattern_docomo))
#    pattern_kddi = re.compile(u"[%s]" % u''.join(pattern_kddi))
#    pattern_softbank = re.compile(u"[%s]" % u''.join(pattern_softbank))
    pattern_unicode = re.compile(u"%s" % u'|'.join(pattern_unicode))
    pattern_google = re.compile(u"%s" % u'|'.join(pattern_google))
    pattern_docomo = re.compile(u"%s" % u'|'.join(pattern_docomo))
    pattern_kddi = re.compile(u"%s" % u'|'.join(pattern_kddi))
    pattern_softbank = re.compile(u"%s" % u'|'.join(pattern_softbank))
    return {
        #                forward            reverse
        'text':         (None,              pattern_unicode),
        'docomo_img':   (None,              pattern_unicode),
        'kddi_img':     (None,              pattern_unicode),
        'softbank_img': (None,              pattern_unicode),
        'google':       (pattern_google,    pattern_unicode),
        'docomo':       (pattern_docomo,    pattern_unicode),
        'kddi':         (pattern_kddi,      pattern_unicode),
        'softbank':     (pattern_softbank,  pattern_unicode),
    }