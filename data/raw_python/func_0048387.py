def xor(self, txt, key):
        """
        >>> crypted = XorCryptor().xor(b"1234", b"ABCD")
        >>> crypted == b'pppp'
        True
        >>> txt = XorCryptor().xor(b'pppp', b"ABCD")
        >>> txt == b'1234'
        True
        """
        assert isinstance(txt, six.binary_type), "txt: %s is not binary type!" % repr(txt)
        assert isinstance(key, six.binary_type), "key: %s is not binary type!" % repr(key)

        if len(txt) != len(key):
            raise SecureJSLoginError("XOR cipher error: '%s' and '%s' must have the same length!" % (txt, key))

        if six.PY2:
            crypted = "".join([chr(ord(t) ^ ord(k)) for t, k in zip(txt, key)])
        else:
            crypted = [(t ^ k) for t, k in zip(txt, key)]
            crypted = bytes(crypted)
        # log.debug("xor(txt='%s', key='%s'): '%s'", txt, key, crypted)
        return crypted