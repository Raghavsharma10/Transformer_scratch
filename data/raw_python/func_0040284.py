def inline(self) -> str:
        """
        Return endpoint string

        :return:
        """
        inlined = [str(info) for info in (self.server, self.port) if info]
        return ESSubscribtionEndpoint.API + " " + " ".join(inlined)