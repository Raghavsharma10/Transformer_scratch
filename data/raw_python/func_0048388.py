def encrypt(self, txt, key):
        """
        XOR ciphering with a PBKDF2 checksum
        """
        # log.debug("encrypt(txt='%s', key='%s')", txt, key)
        assert isinstance(txt, six.text_type), "txt: %s is not text type!" % repr(txt)
        assert isinstance(key, six.text_type), "key: %s is not text type!" % repr(key)

        if len(txt) != len(key):
            raise SecureJSLoginError("encrypt error: %s and '%s' must have the same length!" % (txt, key))

        pbkdf2_hash = PBKDF2SHA1Hasher1().get_salt_hash(txt)

        txt=force_bytes(txt)
        key=force_bytes(key)
        crypted = self.xor(txt, key)
        crypted = binascii.hexlify(crypted)
        crypted = six.text_type(crypted, "ascii")
        return "%s$%s" % (pbkdf2_hash, crypted)