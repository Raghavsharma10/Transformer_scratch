def _parse_text(self, text):
        """Parse text (string) and return list of parsed sentences (strings).

        Each sentence consists of space separated token elements and the
        token format returned by the PatternParser is WORD/TAG/PHRASE/ROLE/(LEMMA)
        (separated by a forward slash '/')

        :param str text: A string.

        """
        if isinstance(self.tokenizer, PatternTokenizer):
            parsed_text = pattern_parse(text, tokenize=True, lemmata=False)
        else:
            _tokenized = []
            _sentences = sent_tokenize(text, tokenizer=self.tokenizer)
            for s in _sentences:
                _tokenized.append(" ".join(self.tokenizer.tokenize(s)))
            parsed_text = pattern_parse(
                _tokenized,
                tokenize=False,
                lemmata=False)
        return parsed_text.split('\n')