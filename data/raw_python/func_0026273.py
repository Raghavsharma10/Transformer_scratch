def generate_feature_matrix(self, mode='tfidf'):
        """
        Returns a feature matrix in the form of a list of lists which
        represents the terms and documents in this Inverted Index using
        the tf-idf weighting by default. The term counts in each
        document can alternatively be used by specifying scheme='count'.

        A custom weighting function can also be passed which receives a term
        and document as parameters.

        The size of the matrix is equal to m x n where m is
        the number of documents and n is the number of terms.

        The list-of-lists format returned by this function can be very easily
        converted to a numpy matrix if required using the `np.as_matrix`
        method.
        """
        result = []

        for doc in self._documents:
            result.append(self.generate_document_vector(doc, mode))

        return result