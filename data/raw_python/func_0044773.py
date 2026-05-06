def _get_tokens_dict(self):
        """
        Accesses tokens dict

        :return: The ordered dict of the tokens
        :rtype: collections.OrderedDict

        """
        if self._tokens_dict is None:
            tokens = [Token(element) for element in self._element.xpath('tokens/token')]
            self._tokens_dict = OrderedDict([(t.id, t) for t in tokens])
        return self._tokens_dict