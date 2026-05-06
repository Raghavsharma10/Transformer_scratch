def code_to_sjis(code):
    u"""Convert character code(hex) to string"""
    if code and isinstance(code, basestring):
        clean_code = code.replace('>', '')
        if clean_code:
            _code_to_sjis_char = lambda c: ''.join([chr(int("%c%c"%(a, b), 16)) for a, b in izip(c[0::2], c[1::2])])
            return ''.join([_code_to_sjis_char(code_char) for code_char in clean_code.split('+') if code_char])
    return None