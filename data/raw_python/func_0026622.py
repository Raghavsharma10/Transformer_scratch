def findall(self, string):
        """ Parse string, returning all outputs as parsed by functions
        """
        output = []
        for match in self.pattern.findall(string):
            if hasattr(match, 'strip'):
                match = [match]
            self._list_add(output, self.run(match))
        return output