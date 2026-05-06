def raw(self) -> str:
        """
        Return signed raw format string of the Membership instance

        :return:
        """
        return """Version: {0}
Type: Membership
Currency: {1}
Issuer: {2}
Block: {3}
Membership: {4}
UserID: {5}
CertTS: {6}
""".format(self.version,
           self.currency,
           self.issuer,
           self.membership_ts,
           self.membership_type,
           self.uid,
           self.identity_ts)