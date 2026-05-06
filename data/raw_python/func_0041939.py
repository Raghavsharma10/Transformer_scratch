def token(cls: Type[SIGType], pubkey: str) -> SIGType:
        """
        Return SIG instance from pubkey

        :param pubkey: Public key of the signature issuer
        :return:
        """
        sig = cls()
        sig.pubkey = pubkey
        return sig