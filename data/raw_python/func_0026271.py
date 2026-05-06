def get_tfidf(self, term, document, normalized=False):
        """
        Returns the Term-Frequency Inverse-Document-Frequency value for the given
        term in the specified document. If normalized is True, term frequency will
        be divided by the document length.
        """
        tf = self.get_term_frequency(term, document)

        # Speeds up performance by avoiding extra calculations
        if tf != 0.0:
            # Add 1 to document frequency to prevent divide by 0
            # (Laplacian Correction)
            df = 1 + self.get_document_frequency(term)
            n = 2 + len(self._documents)

            if normalized:
                tf /= self.get_document_length(document)

            return tf * math.log10(n / df)
        else:
            return 0.0