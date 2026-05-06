def from_seedhex_file(path: str) -> SigningKeyType:
        """
        Return SigningKey instance from Seedhex file

        :param str path: Hexadecimal seed file path
        """
        with open(path, 'r') as fh:
            seedhex = fh.read()
        return SigningKey.from_seedhex(seedhex)