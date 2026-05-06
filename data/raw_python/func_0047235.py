def _decrypt(self, fp, password=None):
        """
        Internal decryption function

        Uses either the password argument for the decryption,
        or, if not supplied, the password field of the object

        :param fp: a file object or similar which supports the readline and read methods
        :rtype: Proxy
        """
        if AES is None:
            raise ImportError("PyCrypto required")
        
        if password is None:
            password = self.password

        if password is None:
            raise ValueError(
                "Password need to be provided to extract encrypted archives")

        # read the PBKDF2 parameters
        # salt
        user_salt = fp.readline().strip()
        user_salt = binascii.a2b_hex(user_salt)
        # checksum salt
        ck_salt = fp.readline().strip()
        ck_salt = binascii.a2b_hex(ck_salt)
        # hashing rounds
        rounds = fp.readline().strip()
        rounds = int(rounds)
        # encryption IV
        iv = fp.readline().strip()
        iv = binascii.a2b_hex(iv)
        # encrypted master key
        master_key = fp.readline().strip()
        master_key = binascii.a2b_hex(master_key)

        # generate key for decrypting the master key
        user_key = PBKDF2(password, user_salt, dkLen=256 // 8, count=rounds)
        # decrypt the master key and iv
        cipher = AES.new(user_key,
                         mode=AES.MODE_CBC,
                         IV=iv)
        master_key = bytearray(cipher.decrypt(master_key))
        # format: <len IV: 1 byte><IV: n bytes><len key: 1 byte><key: m bytes><len checksum: 1 byte><checksum: k bytes>
        # get IV
        l = master_key.pop(0)
        master_iv = bytes(master_key[:l])
        master_key = master_key[l:]
        # get key
        l = master_key.pop(0)
        mk = bytes(master_key[:l])
        master_key = master_key[l:]
        # get checksum
        l = master_key.pop(0)
        master_ck = bytes(master_key[:l])

        # double encode utf8
        utf8mk = self.encode_utf8(mk)
        # calculate checksum by using PBKDF2
        calc_ck = PBKDF2(utf8mk, ck_salt, dkLen=256//8, count=rounds)
        assert calc_ck == master_ck
        # install decryption key
        cipher = AES.new(mk,
                         mode=AES.MODE_CBC,
                         IV=master_iv)

        off = fp.tell()
        fp.seek(0, 2)
        length = fp.tell() - off
        fp.seek(off)

        if self.stream:
            # decryption transformer for Proxy class
            def decrypt(data):
                data = bytearray(cipher.decrypt(data))

                if fp.tell() - off >= length:
                    # check padding (PKCS#7)
                    pad = data[-1]
                    assert data.endswith(bytearray([pad] * pad)), "Expected {!r} got {!r}".format(bytearray([pad] * pad), data[-pad:])
                    data = data[:-pad]

                return data

            return Proxy(decrypt, fp, cipher.block_size)
        else:
            data = bytearray(cipher.decrypt(fp.read()))
            pad = data[-1]
            assert data.endswith(bytearray([pad] * pad)), "Expected {!r} got {!r}".format(bytearray([pad] * pad), data[-pad:])
            data = data[:-pad]
            return io.BytesIO(data)