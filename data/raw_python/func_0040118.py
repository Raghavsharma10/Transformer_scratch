def raw(self) -> str:
        """
        Return a raw format string of the Peer document

        :return:
        """
        doc = """Version: {0}
Type: Peer
Currency: {1}
PublicKey: {2}
Block: {3}
Endpoints:
""".format(self.version, self.currency, self.pubkey, self.blockUID)

        for _endpoint in self.endpoints:
            doc += "{0}\n".format(_endpoint.inline())

        return doc