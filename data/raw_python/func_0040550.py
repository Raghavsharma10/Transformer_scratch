def from_ewif_hex(cls: Type[SigningKeyType], ewif_hex: str, password: str) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter EWIF in hexadecimal format

        :param ewif_hex: EWIF string in hexadecimal format
        :param password: Password of the encrypted seed
        """
        ewif_bytes = Base58Encoder.decode(ewif_hex)
        if len(ewif_bytes) != 39:
            raise Exception("Error: the size of EWIF is invalid")

        # extract data
        fi = ewif_bytes[0:1]
        checksum_from_ewif = ewif_bytes[-2:]
        ewif_no_checksum = ewif_bytes[0:-2]
        salt = ewif_bytes[1:5]
        encryptedhalf1 = ewif_bytes[5:21]
        encryptedhalf2 = ewif_bytes[21:37]

        # check format flag
        if fi != b"\x02":
            raise Exception("Error: bad format version, not EWIF")

        # checksum control
        checksum = libnacl.crypto_hash_sha256(libnacl.crypto_hash_sha256(ewif_no_checksum))[0:2]
        if checksum_from_ewif != checksum:
            raise Exception("Error: bad checksum of the EWIF")

        # SCRYPT
        password_bytes = password.encode("utf-8")
        scrypt_seed = scrypt(password_bytes, salt, 16384, 8, 8, 64)
        derivedhalf1 = scrypt_seed[0:32]
        derivedhalf2 = scrypt_seed[32:64]

        # AES
        aes = pyaes.AESModeOfOperationECB(derivedhalf2)
        decryptedhalf1 = aes.decrypt(encryptedhalf1)
        decryptedhalf2 = aes.decrypt(encryptedhalf2)

        # XOR
        seed1 = xor_bytes(decryptedhalf1, derivedhalf1[0:16])
        seed2 = xor_bytes(decryptedhalf2, derivedhalf1[16:32])
        seed = bytes(seed1 + seed2)

        # Password Control
        signer = SigningKey(seed)
        salt_from_seed = libnacl.crypto_hash_sha256(
            libnacl.crypto_hash_sha256(
                Base58Encoder.decode(signer.pubkey)))[0:4]
        if salt_from_seed != salt:
            raise Exception("Error: bad Password of EWIF address")

        return cls(seed)