def inline(self) -> str:
        """
        Return endpoint string

        :return:
        """
        inlined = [str(info) for info in (self.ws2pid, self.server, self.port, self.path) if info]
        return WS2PEndpoint.API + " " + " ".join(inlined)