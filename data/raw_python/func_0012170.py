def transform(self, corpus, remove_oov=False, norm=False):
        """
        Transform a corpus by repeated calls to vectorize, defined above.

        Parameters
        ----------
        corpus : A list of strings, list of list of strings.
            Represents a corpus as a list of sentences, where sentences
            can either be strings or lists of tokens.
        remove_oov : bool, optional, default False
            If True, removes OOV items from the input before vectorization.

        Returns
        -------
        c : list
            A list of numpy arrays, where each array represents the transformed
            sentence in the original list. The list is guaranteed to be the
            same length as the input list, but the arrays in the list may be
            of different lengths, depending on whether remove_oov is True.

        """
        return [self.vectorize(s, remove_oov=remove_oov, norm=norm)
                for s in corpus]