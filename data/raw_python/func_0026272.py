def generate_document_vector(self, doc, mode='tfidf'):
        """
        Returns a representation of the specified document as a feature vector
        weighted according the mode specified (by default tf-dif).

        A custom weighting function can also be passed which receives the hashedindex
        instance, the selected term and document as parameters.

        The result will be returned in the form of a list. This can be converted
        into a numpy array if required using the `np.asarray` method
        Available built-in modes:
          * tfidf: Term Frequency Inverse Document Frequency
          * ntfidf: Normalized Term Frequency Inverse Document Frequency
          * tf: Term Frequency
          * ntf: Normalized Term Frequency
        """
        if mode == 'tfidf':
            selected_function = HashedIndex.get_tfidf
        elif mode == 'ntfidf':
            selected_function = functools.partial(HashedIndex.get_tfidf, normalized=True)
        elif mode == 'tf':
            selected_function = HashedIndex.get_term_frequency
        elif mode == 'ntf':
            selected_function = functools.partial(HashedIndex.get_term_frequency, normalized=True)
        elif hasattr(mode, '__call__'):
            selected_function = mode
        else:
            raise ValueError('Unexpected mode: %s', mode)

        result = []
        for term in self._terms:
            result.append(selected_function(self, term, doc))

        return result