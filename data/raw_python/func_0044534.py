def sentence(self):
        """
        The sentence related to this mention

        :getter: returns the sentence this mention relates to
        :type: corenlp_xml.document.Sentence

        """
        if self._sentence is None:
            sentences = self._element.xpath('sentence/text()')
            if len(sentences) > 0:
                self._sentence = self._coref.document.get_sentence_by_id(int(sentences[0]))
        return self._sentence