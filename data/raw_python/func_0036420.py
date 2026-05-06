def vectorize(self, docs):
        """
        Vectorizes a list of documents using their DCS representations.
        """
        doc_core_sems, all_concepts = self._extract_core_semantics(docs)

        shape = (len(docs), len(all_concepts))
        vecs = np.zeros(shape)
        for i, core_sems in enumerate(doc_core_sems):
            for con, weight in core_sems:
                j = all_concepts.index(con)
                vecs[i,j] = weight

        # Normalize
        return vecs/np.max(vecs)