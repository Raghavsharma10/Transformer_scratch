def from_wif_or_ewif_file(path: str, password: Optional[str] = None) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter WIF or EWIF file

        :param path: Path to WIF of EWIF file
        :param password: Password needed for EWIF file
        """
        with open(path, 'r') as fh:
            wif_content = fh.read()

        # check data field
        regex = compile('Data: ([1-9A-HJ-NP-Za-km-z]+)', MULTILINE)
        match = search(regex, wif_content)
        if not match:
            raise Exception('Error: Bad format WIF or EWIF v1 file')

        # capture hexa wif key
        wif_hex = match.groups()[0]
        return SigningKey.from_wif_or_ewif_hex(wif_hex, password)