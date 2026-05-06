def _parse_text(self, text):
        """Parse text (string) and return list of parsed sentences (strings).

        Each sentence consists of space separated token elements and the
        token format returned by the PatternParser is WORD/TAG/PHRASE/ROLE/LEMMA
        (separated by a forward slash '/')

        :param str text: A string.

        """
        # Fix for issue #1
        text = text.replace("/", " FORWARDSLASH ")
        _tokenized = " ".join(self.tokenizer.tokenize(text))
        parsed_text = pattern_parse(_tokenized, tokenize=False, lemmata=True)
        return parsed_text.split('\n')