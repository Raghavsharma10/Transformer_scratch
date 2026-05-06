def _encode_str(self, obj, escape_quotes=True):
        """Return an ASCII-only JSON representation of a Python string"""
        def replace(match):
            s = match.group(0)
            try:
                if escape_quotes:
                    return ESCAPE_DCT[s]
                else:
                    return BASE_ESCAPE_DCT[s]
            except KeyError:
                n = ord(s)
                if n < 0x10000:
                    return '\\u{0:04x}'.format(n)
                else:
                    # surrogate pair
                    n -= 0x10000
                    s1 = 0xd800 | ((n >> 10) & 0x3ff)
                    s2 = 0xdc00 | (n & 0x3ff)
                    return '\\u{0:04x}\\u{1:04x}'.format(s1, s2)
        if escape_quotes:
            return '"' + ESCAPE_ASCII.sub(replace, obj) + '"'
        else:
            return BASE_ESCAPE_ASCII.sub(replace, obj)