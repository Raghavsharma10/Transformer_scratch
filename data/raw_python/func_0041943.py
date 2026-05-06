def token(cls: Type[XHXType], sha_hash: str) -> XHXType:
        """
        Return XHX instance from sha_hash

        :param sha_hash: SHA256 hash
        :return:
        """
        xhx = cls()
        xhx.sha_hash = sha_hash
        return xhx