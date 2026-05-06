def from_seedhex(cls: Type[SigningKeyType], seedhex: str) -> SigningKeyType:
        """
        Return SigningKey instance from Seedhex

        :param str seedhex: Hexadecimal seed string
        """
        regex_seedhex = compile("([0-9a-fA-F]{64})")
        match = search(regex_seedhex, seedhex)
        if not match:
            raise Exception('Error: Bad seed hexadecimal format')
        seedhex = match.groups()[0]
        seed = convert_seedhex_to_seed(seedhex)
        return cls(seed)