def inline(self) -> str:
        """
        Return inline document string

        :return:
        """
        return "{0}:{1}:{2}:{3}".format(self.pubkey_from, self.pubkey_to,
                                        self.timestamp.number, self.signatures[0])