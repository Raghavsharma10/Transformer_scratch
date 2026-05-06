def encode_utf8(mk):
        """
        (Double-)encodes the given string (masterkey) with utf-8

        Tries to behave like the Java implementation
        """
        utf8mk = mk.decode('raw_unicode_escape')
        utf8mk = list(utf8mk)
        to_char = chr
        if sys.version_info[0] < 3:
            to_char = unichr
        for i in range(len(utf8mk)):
            c = ord(utf8mk[i])
            # fix java encoding (add 0xFF00 to non ascii chars)
            if 0x7f < c < 0x100:
                c += 0xff00
                utf8mk[i] = to_char(c)
        return ''.join(utf8mk).encode('utf-8')