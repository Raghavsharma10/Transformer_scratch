def save_ewif_file(self, path: str, password: str) -> None:
        """
        Save an Encrypted Wallet Import Format file (WIF v2)

        :param path: Path to file
        :param password:
        """
        # version
        version = 1

        # add version to seed
        salt = libnacl.crypto_hash_sha256(
            libnacl.crypto_hash_sha256(
                Base58Encoder.decode(self.pubkey)))[0:4]

        # SCRYPT
        password_bytes = password.encode("utf-8")
        scrypt_seed = scrypt(password_bytes, salt, 16384, 8, 8, 64)
        derivedhalf1 = scrypt_seed[0:32]
        derivedhalf2 = scrypt_seed[32:64]

        # XOR
        seed1_xor_derivedhalf1_1 = bytes(xor_bytes(self.seed[0:16], derivedhalf1[0:16]))
        seed2_xor_derivedhalf1_2 = bytes(xor_bytes(self.seed[16:32], derivedhalf1[16:32]))

        # AES
        aes = pyaes.AESModeOfOperationECB(derivedhalf2)
        encryptedhalf1 = aes.encrypt(seed1_xor_derivedhalf1_1)
        encryptedhalf2 = aes.encrypt(seed2_xor_derivedhalf1_2)

        # add format to final seed (1=WIF,2=EWIF)
        seed_bytes = b'\x02' + salt + encryptedhalf1 + encryptedhalf2

        # calculate checksum
        sha256_v1 = libnacl.crypto_hash_sha256(seed_bytes)
        sha256_v2 = libnacl.crypto_hash_sha256(sha256_v1)
        checksum = sha256_v2[0:2]

        # B58 encode final key string
        ewif_key = Base58Encoder.encode(seed_bytes + checksum)

        # save file
        with open(path, 'w') as fh:
            fh.write(
                """Type: EWIF
Version: {version}
Data: {data}""".format(version=version, data=ewif_key)
            )