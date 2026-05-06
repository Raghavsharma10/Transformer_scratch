def tokenize(self, docs):
        """ Tokenizes a document, using a lemmatizer.

        Args:
            | doc (str)                 -- the text document to process.

        Returns:
            | list                      -- the list of tokens.
        """
        if self.n_jobs == 1:
            return [self._tokenize(doc) for doc in docs]
        else:
            return parallel(self._tokenize, docs, self.n_jobs)