def _tokenize(cls, sentence):
        """
        Split a sentence while preserving tags.
        """
        while True:
            match = cls._regex_tag.search(sentence)
            if not match:
                yield from cls._split(sentence)
                return
            chunk = sentence[:match.start()]
            yield from cls._split(chunk)
            tag = match.group(0)
            yield tag
            sentence = sentence[(len(chunk) + len(tag)):]