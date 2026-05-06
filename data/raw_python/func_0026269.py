def get_document_length(self, document):
        """
        Returns the number of terms found within the specified document.
        """
        if document in self._documents:
            return self._documents[document]
        else:
            raise IndexError(DOCUMENT_DOES_NOT_EXIST)