def _detailTuples(self, uriRefs):
        """
        Given a list of uriRefs, return a list of dicts:
        {'subject': s, 'predicate': p, 'object': o }
        all values are strings
        """
        details = []
        for uriRef in uriRefs:
            for subject, predicate, object_ in self._rdfGraph.triples(
                    (uriRef, None, None)):
                details.append({
                    'subject': subject.toPython(),
                    'predicate': predicate.toPython(),
                    'object': object_.toPython()
                })
        return details