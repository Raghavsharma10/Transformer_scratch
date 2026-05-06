def inline(self) -> str:
        """
        Return endpoint string

        :return:
        """
        inlined = [str(info) for info in (self.server, self.ipv4, self.ipv6, self.port, self.path) if info]
        return SecuredBMAEndpoint.API + " " + " ".join(inlined)