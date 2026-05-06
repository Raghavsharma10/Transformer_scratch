def extract(self, text):
        """Return a list of noun phrases (strings) for a body of text.

        :param str text: A string.

        """
        _extracted = []
        if text.strip() == "":
            return _extracted
        parsed_sentences = self._parse_text(text)
        for s in parsed_sentences:
            tokens = s.split()
            new_np = []
            for t in tokens:
                w, tag, phrase, role = t.split('/')
                # exclude some parser errors (e.g. VB within NP),
                # extend startswith tuple if necessary
                if 'NP' in phrase and not self._is_verb(w, tag):
                    if len(new_np) > 0 and w.lower() in START_NEW_NP:
                        _extracted.append(" ".join(new_np))
                        new_np = [w]
                    else:
                        # normalize capitalisation of sentence starters, except
                        # for nouns
                        new_np.append(w.lower() if tokens[0].startswith(w) and
                                      not tag.startswith('N') else w)
                else:
                    if len(new_np) > 0:
                        _extracted.append(" ".join(new_np))
                    new_np = []
        return self._filter_extracted(_extracted)