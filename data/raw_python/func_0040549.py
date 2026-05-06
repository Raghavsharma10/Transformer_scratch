def from_ewif_file(path: str, password: str) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter EWIF file

        :param path: Path to EWIF file
        :param password: Password of the encrypted seed
        """
        with open(path, 'r') as fh:
            wif_content = fh.read()

        # check data field
        regex = compile('Data: ([1-9A-HJ-NP-Za-km-z]+)', MULTILINE)
        match = search(regex, wif_content)
        if not match:
            raise Exception('Error: Bad format EWIF v1 file')

        # capture ewif key
        ewif_hex = match.groups()[0]
        return SigningKey.from_ewif_hex(ewif_hex, password)