def _encrypt(self, dec, password=None):
        """
        Internal encryption function

        Uses either the password argument for the encryption,
        or, if not supplied, the password field of the object

        :param dec: a byte string representing the to be encrypted data
        :rtype: bytes
        """
        if AES is None:
            raise ImportError("PyCrypto required")

        if password is None:
            password = self.password

        if password is None:
            raise ValueError(
                "Password need to be provided to create encrypted archives")

        # generate the different encryption parts (non-secure!)
        master_key = Random.get_random_bytes(32)
        master_salt = Random.get_random_bytes(64)
        user_salt = Random.get_random_bytes(64)
        master_iv = Random.get_random_bytes(16)
        user_iv = Random.get_random_bytes(16)
        rounds = 10000

        # create the PKCS#7 padding
        l = len(dec)
        pad = 16 - (l % 16)
        dec += bytes([pad] * pad)

        # encrypt the data
        cipher = AES.new(master_key, IV=master_iv, mode=AES.MODE_CBC)
        enc = cipher.encrypt(dec)

        # generate the master key checksum
        master_ck = PBKDF2(self.encode_utf8(master_key),
                           master_salt, dkLen=256//8, count=rounds)

        # generate the user key from the given password
        user_key = PBKDF2(password,
                          user_salt, dkLen=256//8, count=rounds)

        # encrypt the master key and iv
        master_dec = b"\x10" + master_iv + b"\x20" + master_key + b"\x20" + master_ck
        l = len(master_dec)
        pad = 16 - (l % 16)
        master_dec += bytes([pad] * pad)
        cipher = AES.new(user_key, IV=user_iv, mode=AES.MODE_CBC)
        master_enc = cipher.encrypt(master_dec)

        # put everything together
        enc = binascii.b2a_hex(user_salt).upper() + b"\n" + \
                binascii.b2a_hex(master_salt).upper() + b"\n" + \
                str(rounds).encode() + b"\n" + \
                binascii.b2a_hex(user_iv).upper() + b"\n" + \
                binascii.b2a_hex(master_enc).upper() + b"\n" + enc

        return enc