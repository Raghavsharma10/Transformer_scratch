def save_wif_file(self, path: str) -> None:
        """
        Save a Wallet Import Format file (WIF) v1

        :param path: Path to file
        """
        # version
        version = 1

        # add format to seed (1=WIF,2=EWIF)
        seed_fi = b"\x01" + self.seed

        # calculate checksum
        sha256_v1 = libnacl.crypto_hash_sha256(seed_fi)
        sha256_v2 = libnacl.crypto_hash_sha256(sha256_v1)
        checksum = sha256_v2[0:2]

        # base58 encode key and checksum
        wif_key = Base58Encoder.encode(seed_fi + checksum)

        with open(path, 'w') as fh:
            fh.write(
                """Type: WIF
Version: {version}
Data: {data}""".format(version=version, data=wif_key)
            )