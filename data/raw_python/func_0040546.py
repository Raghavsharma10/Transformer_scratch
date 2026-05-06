def from_wif_file(path: str) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter WIF file

        :param path: Path to WIF file
        """
        with open(path, 'r') as fh:
            wif_content = fh.read()

        # check data field
        regex = compile('Data: ([1-9A-HJ-NP-Za-km-z]+)', MULTILINE)
        match = search(regex, wif_content)
        if not match:
            raise Exception('Error: Bad format WIF v1 file')

        # capture hexa wif key
        wif_hex = match.groups()[0]
        return SigningKey.from_wif_hex(wif_hex)