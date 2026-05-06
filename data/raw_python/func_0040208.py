def inline(self) -> str:
        """
        Return inline string format of the Membership instance
        :return:
        """
        return "{0}:{1}:{2}:{3}:{4}".format(self.issuer,
                                            self.signatures[0],
                                            self.membership_ts,
                                            self.identity_ts,
                                            self.uid)