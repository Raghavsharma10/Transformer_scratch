def sha_hash(self) -> str:
        """
        Return uppercase hex sha256 hash from signed raw document

        :return:
        """
        return hashlib.sha256(self.signed_raw().encode("ascii")).hexdigest().upper()