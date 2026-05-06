def parse(self, text):
        """Parses the text.

        ``pattern.de.parse(**kwargs)`` can be passed to the parser instance and
        are documented in the main docstring of
        :class:`PatternParser() <textblob_de.parsers.PatternParser>`.

        :param str text: A string.

        """
        #: Do not process empty strings (Issue #3)
        if text.strip() == "":
            return ""
        #: Do not process strings consisting of a single punctuation mark (Issue #4)
        elif text.strip() in PUNCTUATION:
            _sym = text.strip()
            if _sym in tuple('.?!'):
                _tag = "."
            else:
                _tag = _sym
            if self.lemmata:
                return "{0}/{1}/O/O/{0}".format(_sym, _tag)
            else:
                return "{0}/{1}/O/O".format(_sym, _tag)
        if self.tokenize:
            _tokenized = " ".join(self.tokenizer.tokenize(text))
        else:
            _tokenized = text

        _parsed = pattern_parse(_tokenized,
                                # text is tokenized before it is passed on to
                                # pattern.de.parse
                                tokenize=False,
                                tags=self.tags, chunks=self.chunks,
                                relations=self.relations, lemmata=self.lemmata,
                                encoding=self.encoding, tagset=self.tagset)
        if self.pprint:
            _parsed = pattern_pprint(_parsed)

        return _parsed