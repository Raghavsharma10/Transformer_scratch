def _get_sentences_dict(self):
        """
        Returns sentence objects

        :return: order dict of sentences
        :rtype: collections.OrderedDict

        """
        if self._sentences_dict is None:
            sentences = [Sentence(element) for element in self._xml.xpath('/root/document/sentences/sentence')]
            self._sentences_dict = OrderedDict([(s.id, s) for s in sentences])
        return self._sentences_dict