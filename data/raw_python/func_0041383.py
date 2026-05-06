def compact(self) -> str:
        """
        Return a transaction in its compact format from the instance

        :return:
        """
        """TX:VERSION:NB_ISSUERS:NB_INPUTS:NB_UNLOCKS:NB_OUTPUTS:HAS_COMMENT:LOCKTIME
PUBLIC_KEY:INDEX
...
INDEX:SOURCE:FINGERPRINT:AMOUNT
...
PUBLIC_KEY:AMOUNT
...
COMMENT
"""
        doc = "TX:{0}:{1}:{2}:{3}:{4}:{5}:{6}\n".format(self.version,
                                                        len(self.issuers),
                                                        len(self.inputs),
                                                        len(self.unlocks),
                                                        len(self.outputs),
                                                        '1' if self.comment != "" else '0',
                                                        self.locktime)
        if self.version >= 3:
            doc += "{0}\n".format(self.blockstamp)

        for pubkey in self.issuers:
            doc += "{0}\n".format(pubkey)
        for i in self.inputs:
            doc += "{0}\n".format(i.inline(self.version))
        for u in self.unlocks:
            doc += "{0}\n".format(u.inline())
        for o in self.outputs:
            doc += "{0}\n".format(o.inline())
        if self.comment != "":
            doc += "{0}\n".format(self.comment)
        for s in self.signatures:
            doc += "{0}\n".format(s)

        return doc