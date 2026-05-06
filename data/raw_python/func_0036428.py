def _extract_core_semantics(self, docs):
        """
        Extracts core semantics for a list of documents, returning them along with
        a list of all the concepts represented.
        """
        all_concepts = []
        doc_core_sems = []
        for doc in docs:
            core_sems = self._process_doc(doc)
            doc_core_sems.append(core_sems)
            all_concepts += [con for con, weight in core_sems]
        return doc_core_sems, list(set(all_concepts))