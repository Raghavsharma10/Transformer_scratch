def lemmatize(self):
        """Return the lemma of each word in this WordList.

        Currently using NLTKPunktTokenizer() for all lemmatization
        tasks. This might cause slightly different tokenization results
        compared to the TextBlob.words property.

        """
        _lemmatizer = PatternParserLemmatizer(tokenizer=NLTKPunktTokenizer())
        # WordList object --> Sentence.string
        # add a period (improves parser accuracy)
        _raw = " ".join(self) + "."
        _lemmas = _lemmatizer.lemmatize(_raw)
        return self.__class__([Word(l, t) for l, t in _lemmas])