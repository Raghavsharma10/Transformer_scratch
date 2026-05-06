def translate_char(source_char, carrier, reverse=False, encoding=False):
    u"""translate unicode emoji character to unicode carrier emoji character (or reverse)
    
    Attributes:
        source_char   - emoji character. it must be unicode instance or have to set `encoding` attribute to decode
        carrier       - the target carrier
        reverse       - if you want to translate CARRIER => UNICODE, turn it True
        encoding      - encoding name for decode (Default is None)
    
    """
    if not isinstance(source_char, unicode) and encoding:
        source_char = source_char.decode(encoding, 'replace')
    elif not isinstance(source_char, unicode):
        raise AttributeError(u"`source_char` must be decoded to `unicode` or set `encoding` attribute to decode `source_char`")
    if len(source_char) > 1:
        raise AttributeError(u"`source_char` must be a letter. use `translate` method insted.")
    translate_dictionary = _loader.translate_dictionaries[carrier]
    if not reverse:
        translate_dictionary = translate_dictionary[0]
    else:
        translate_dictionary = translate_dictionary[1]
    if not translate_dictionary:
        return source_char
    return translate_dictionary.get(source_char, source_char)