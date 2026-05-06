def create_translate_dictionaries(symbols):
    u"""create translate dictionaries for text, google, docomo, kddi and softbank via `symbols`
    
    create dictionaries for translate emoji character to carrier from unicode (forward) or to unicode from carrier (reverse).
    method return dictionary instance which key is carrier name and value format is `(forward_dictionary, reverse_dictionary)`
    each dictionary expect `unicode` format. any text not decoded have to be decode before using this dictionary (like matching key)
    
    DO NOT CONFUSE with carrier's UNICODE emoji. UNICODE emoji like `u"\uE63E"` for DoCoMo's sun emoji is not expected. expected character
    for DoCoMo's sun is decoded character from `"\xF8\x9F"` (actually decoded unicode of `"\xF8\xF9"` is `u"\uE63E"` however not all emoji
    can convert with general encode/decode method. conversion of UNICODE <-> ShiftJIS is operated in Symbol constructor and stored in Symbol's `sjis`
    attribute and unicode formatted is `usjis` attribute.)
        
    """
    unicode_to_text = {}
    unicode_to_docomo_img = {}
    unicode_to_kddi_img = {}
    unicode_to_softbank_img = {}
    unicode_to_google = {}
    unicode_to_docomo = {}
    unicode_to_kddi = {}
    unicode_to_softbank = {}
    google_to_unicode = {}
    docomo_to_unicode = {}
    kddi_to_unicode = {}
    softbank_to_unicode = {}
    for x in symbols:
        if x.unicode.keyable:
            unicode_to_text[unicode(x.unicode)] = x.unicode.fallback
            unicode_to_docomo_img[unicode(x.unicode)] = x.docomo.thumbnail
            unicode_to_kddi_img[unicode(x.unicode)] = x.kddi.thumbnail
            unicode_to_softbank_img[unicode(x.unicode)] = x.softbank.thumbnail
            unicode_to_google[unicode(x.unicode)] = unicode(x.google)
            unicode_to_docomo[unicode(x.unicode)] = unicode(x.docomo)
            unicode_to_kddi[unicode(x.unicode)] = unicode(x.kddi)
            unicode_to_softbank[unicode(x.unicode)] = unicode(x.softbank)
        if x.google.keyable: google_to_unicode[unicode(x.google)] = unicode(x.unicode)
        if x.docomo.keyable: docomo_to_unicode[unicode(x.docomo)] = unicode(x.unicode)
        if x.kddi.keyable: kddi_to_unicode[unicode(x.kddi)] = unicode(x.unicode)
        if x.softbank.keyable: softbank_to_unicode[unicode(x.softbank)] = unicode(x.unicode)
    return {
        #                forward                reverse
        'text':         (None,                  unicode_to_text),
        'docomo_img':   (None,                  unicode_to_docomo_img),
        'kddi_img':     (None,                  unicode_to_kddi_img),
        'softbank_img': (None,                  unicode_to_softbank_img),
        'google':       (google_to_unicode,     unicode_to_google),
        'docomo':       (docomo_to_unicode,     unicode_to_docomo),
        'kddi':         (kddi_to_unicode,       unicode_to_kddi),
        'softbank':     (softbank_to_unicode,   unicode_to_softbank),
    }