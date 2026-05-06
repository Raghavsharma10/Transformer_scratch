def raw(self) -> str:
        """
        Return raw string format from the instance

        :return:
        """
        doc = """Version: {0}
Type: Transaction
Currency: {1}
""".format(self.version,
           self.currency)

        if self.version >= 3:
            doc += "Blockstamp: {0}\n".format(self.blockstamp)

        doc += "Locktime: {0}\n".format(self.locktime)

        doc += "Issuers:\n"
        for p in self.issuers:
            doc += "{0}\n".format(p)

        doc += "Inputs:\n"
        for i in self.inputs:
            doc += "{0}\n".format(i.inline(self.version))

        doc += "Unlocks:\n"
        for u in self.unlocks:
            doc += "{0}\n".format(u.inline())

        doc += "Outputs:\n"
        for o in self.outputs:
            doc += "{0}\n".format(o.inline())

        doc += "Comment: "
        doc += "{0}\n".format(self.comment)

        return doc