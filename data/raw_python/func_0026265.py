def add_term_occurrence(self, term, document):
        """
        Adds an occurrence of the term in the specified document.
        """
        if document not in self._documents:
            self._documents[document] = 0

        if term not in self._terms:
            if self._freeze:
                return
            else:
                self._terms[term] = collections.Counter()

        if document not in self._terms[term]:
            self._terms[term][document] = 0

        self._documents[document] += 1
        self._terms[term][document] += 1