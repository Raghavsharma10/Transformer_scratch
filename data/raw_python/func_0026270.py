def get_documents(self, term):
        """
        Returns all documents related to the specified term in the
        form of a Counter object.
        """
        if term not in self._terms:
            raise IndexError(TERM_DOES_NOT_EXIST)
        else:
            return self._terms[term]