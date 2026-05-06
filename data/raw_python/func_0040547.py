def from_wif_hex(cls: Type[SigningKeyType], wif_hex: str) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter WIF in hexadecimal format

        :param wif_hex: WIF string in hexadecimal format
        """
        wif_bytes = Base58Encoder.decode(wif_hex)
        if len(wif_bytes) != 35:
            raise Exception("Error: the size of WIF is invalid")

        # extract data
        checksum_from_wif = wif_bytes[-2:]
        fi = wif_bytes[0:1]
        seed = wif_bytes[1:-2]
        seed_fi = wif_bytes[0:-2]

        # check WIF format flag
        if fi != b"\x01":
            raise Exception("Error: bad format version, not WIF")

        # checksum control
        checksum = libnacl.crypto_hash_sha256(libnacl.crypto_hash_sha256(seed_fi))[0:2]
        if checksum_from_wif != checksum:
            raise Exception("Error: bad checksum of the WIF")

        return cls(seed)