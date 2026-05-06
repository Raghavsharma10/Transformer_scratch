def code_to_unicode(code):
    u"""Convert character code(hex) to unicode"""
    def utf32chr(n):
        """utf32char for narrow build python"""
        return eval("u\'\\U%08X\'" % n)
    def lazy_unichr(n):
        try:
            return unichr(n)
        except ValueError:
            warnings.warn("Your python it built as narrow python. Make with "
                "'--enable-unicode=ucs4' configure option to build wide python")
            return utf32chr(n)
    if code and isinstance(code, basestring):
        clean_code = code.replace('>', '')
        if clean_code:
            return u''.join([lazy_unichr(int(code_char, 16)) for code_char in clean_code.split('+') if code_char])
    return None