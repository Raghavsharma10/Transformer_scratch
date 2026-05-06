def _formatEvidence(self, elements):
        """
        Formats elements passed into parts of a query for filtering
        """
        elementClause = None
        filters = []
        for evidence in elements:
            if evidence.description:
                elementClause = 'regex(?{}, "{}")'.format(
                    'environment_label', evidence.description)
            if (hasattr(evidence, 'externalIdentifiers') and
                    evidence.externalIdentifiers):
                # TODO will this pick up > 1 externalIdentifiers ?
                for externalIdentifier in evidence['externalIdentifiers']:
                    exid_clause = self._formatExternalIdentifier(
                        externalIdentifier, 'environment')
                    # cleanup parens from _formatExternalIdentifier method
                    elementClause = exid_clause[1:-1]
            if elementClause:
                filters.append(elementClause)
        elementClause = "({})".format(" || ".join(filters))
        return elementClause