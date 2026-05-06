def head(self):
        """
        The token serving as the "head" of the mention

        :getter: the token corresponding to the head
        :type: corenlp_xml.document.Token

        """
        if self._head is None:
            self._head = self.sentence.tokens[self._head_id-1]
        return self._head