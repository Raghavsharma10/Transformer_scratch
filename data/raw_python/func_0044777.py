def parse(self):
        """
        Accesses the parse tree based on the S-expression parse string in the XML

        :getter: Returns the NLTK parse tree
        :type: nltk.Tree

        """
        if self.parse_string is not None and self._parse is None:
            self._parse = Tree.parse(self._parse_string)
        return self._parse