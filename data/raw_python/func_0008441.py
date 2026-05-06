def tag(self, sentence, tokenize=True):
        """Tag a string `sentence`.

        :param str or list sentence: A string or a list of sentence strings.
        :param tokenize: (optional) If ``False`` string has to be tokenized before
            (space separated string).

        """
        #: Do not process empty strings (Issue #3)
        if sentence.strip() == "":
            return []
        #: Do not process strings consisting of a single punctuation mark (Issue #4)
        elif sentence.strip() in PUNCTUATION:
            if self.include_punc:
                _sym = sentence.strip()
                if _sym in tuple('.?!'):
                    _tag = "."
                else:
                    _tag = _sym
                return [(_sym, _tag)]
            else:
                return []
        if tokenize:
            _tokenized = " ".join(self.tokenizer.tokenize(sentence))
            sentence = _tokenized
        # Sentence is tokenized before it is passed on to pattern.de.tag
        # (i.e. it is either submitted tokenized or if )
        _tagged = pattern_tag(sentence, tokenize=False,
                              encoding=self.encoding,
                              tagset=self.tagset)
        if self.include_punc:
            return _tagged
        else:
            _tagged = [
                (word, t) for word, t in _tagged if not PUNCTUATION_REGEX.match(
                    unicode(t))]
            return _tagged