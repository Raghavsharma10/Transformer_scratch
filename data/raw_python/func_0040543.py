def save_pubsec_file(self, path: str) -> None:
        """
        Save a Duniter PubSec file (PubSec) v1

        :param path: Path to file
        """
        # version
        version = 1

        # base58 encode keys
        base58_signing_key = Base58Encoder.encode(self.sk)
        base58_public_key = self.pubkey

        # save file
        with open(path, 'w') as fh:
            fh.write(
                """Type: PubSec
Version: {version}
pub: {pubkey}
sec: {signkey}""".format(version=version, pubkey=base58_public_key, signkey=base58_signing_key)
            )