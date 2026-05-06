def analyze(self, text):
        """Return the sentiment as a tuple of the form:
        ``(polarity, subjectivity)``

        :param str text: A string.

        .. todo::

            Figure out best format to be passed to the analyzer.
            There might be a better format than a string of space separated
            lemmas (e.g. with pos tags) but the parsing/tagging
            results look rather inaccurate and a wrong pos
            might prevent the lexicon lookup of an otherwise correctly
            lemmatized word form (or would it not?) - further checks needed.

        """
        if self.lemmatize:
            text = self._lemmatize(text)
        return self.RETURN_TYPE(*pattern_sentiment(text))