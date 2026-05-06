def scan(self, string):
        """ Like findall, but also returning matching start and end string locations
        """
        return list(self._scanner_to_matches(self.pattern.scanner(string), self.run))