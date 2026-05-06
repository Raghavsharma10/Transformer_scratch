def inline(self) -> str:
        """
        Return endpoint string

        :return:
        """
        doc = self.api
        for p in self.properties:
            doc += " {0}".format(p)
        return doc