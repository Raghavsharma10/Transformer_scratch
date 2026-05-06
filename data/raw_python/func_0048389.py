def decrypt(self, txt, key):
        """
        1. Decrypt a XOR crypted String.
        2. Compare the inserted SHA salt-hash checksum.
        """
        # log.debug("decrypt(txt='%s', key='%s')", txt, key)
        assert isinstance(txt, six.text_type), "txt: %s is not text type!" % repr(txt)
        assert isinstance(key, six.text_type), "key: %s is not text type!" % repr(key)

        pbkdf2_hash, crypted = txt.rsplit("$",1)

        # if not seed_generator.DEBUG and len(pbkdf2_hash)!=SALT_HASH_LEN:
        #     raise SecureJSLoginError(
        #         "encrypt error: Salt-hash %s with length %i must be length %i!" % (
        #             repr(pbkdf2_hash), len(pbkdf2_hash), SALT_HASH_LEN
        #         )
        #     )

        try:
            crypted = binascii.unhexlify(crypted)
        except (binascii.Error, TypeError) as err:
            # Py2 will raise TypeError - Py3 the binascii.Error
            raise SecureJSLoginError("unhexlify error: %s with data: %s" % (err, crypted))

        if len(crypted) != len(key):
            raise SecureJSLoginError("encrypt error: %s and '%s' must have the same length!" % (crypted, key))

        key=force_bytes(key)
        decrypted = self.xor(crypted, key)

        try:
            decrypted = force_text(decrypted)
        except UnicodeDecodeError:
            raise SecureJSLoginError("Can't decode data.")

        test = PBKDF2SHA1Hasher1().verify(decrypted, pbkdf2_hash)
        if not test:
            raise SecureJSLoginError("XOR decrypted data: PBKDF2 hash test failed")

        return decrypted