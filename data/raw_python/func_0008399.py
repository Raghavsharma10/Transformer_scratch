def lemmatize(self, text):
        """Return a list of (lemma, tag) tuples.

        :param str text: A string.

        """
        #: Do not process empty strings (Issue #3)
        if text.strip() == "":
            return []
        parsed_sentences = self._parse_text(text)
        _lemmalist = []
        for s in parsed_sentences:
            tokens = s.split()
            for i, t in enumerate(tokens):
                #: Filter empty tokens from the parser output (Issue #5)
                #: This only happens if parser input is improperly tokenized
                #: e.g. if there are empty strings in the list of tokens ['A', '', '.']
                if t.startswith('/'):
                    continue
                w, tag, phrase, role, lemma = t.split('/')
                # The lexicon uses Swiss spelling: "ss" instead of "ß".
                lemma = lemma.replace(u"ß", "ss")
                # Reverse previous replacement
                lemma = lemma.strip().replace("forwardslash", "/")
                if w[0].isupper() and i > 0:
                    lemma = lemma.title()
                elif tag.startswith("N") and i == 0:
                    lemma = lemma.title()
                # Todo: Check if it makes sense to treat '/' as punctuation
                # (especially for sentiment analysis it might be interesting
                # to treat it as OR ('oder')).
                if w in string.punctuation or lemma == '/':
                    continue
                else:
                    lemma = lemma

                _lemmalist.append((lemma, tag))
        return _lemmalist