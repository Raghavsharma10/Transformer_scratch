def from_pubsec_file(cls: Type[SigningKeyType], path: str) -> SigningKeyType:
        """
        Return SigningKey instance from Duniter WIF file

        :param path: Path to WIF file
        """
        with open(path, 'r') as fh:
            pubsec_content = fh.read()

        # line patterns
        regex_pubkey = compile("pub: ([1-9A-HJ-NP-Za-km-z]{43,44})", MULTILINE)
        regex_signkey = compile("sec: ([1-9A-HJ-NP-Za-km-z]{88,90})", MULTILINE)

        # check public key field
        match = search(regex_pubkey, pubsec_content)
        if not match:
            raise Exception('Error: Bad format PubSec v1 file, missing public key')

        # check signkey field
        match = search(regex_signkey, pubsec_content)
        if not match:
            raise Exception('Error: Bad format PubSec v1 file, missing sec key')

        # capture signkey
        signkey_hex = match.groups()[0]

        # extract seed from signkey
        seed = bytes(Base58Encoder.decode(signkey_hex)[0:32])

        return cls(seed)