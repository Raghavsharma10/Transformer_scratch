def _filterSearchPhenotypesRequest(self, request):
        """
        Filters request for phenotype search requests
        """
        filters = []
        if request.id:
            filters.append("?phenotype = <{}>".format(request.id))

        if request.description:
            filters.append(
                'regex(?phenotype_label, "{}")'.format(request.description))
        # OntologyTerms
        # TODO: refactor this repetitive code
        if hasattr(request.type, 'id') and request.type.id:
            ontolgytermsClause = self._formatOntologyTermObject(
                request.type, 'phenotype')
            if ontolgytermsClause:
                filters.append(ontolgytermsClause)
        if len(request.qualifiers) > 0:
            ontolgytermsClause = self._formatOntologyTermObject(
                request.qualifiers, 'phenotype_quality')
            if ontolgytermsClause:
                filters.append(ontolgytermsClause)
        if hasattr(request.age_of_onset, 'id') and request.age_of_onset.id:
            ontolgytermsClause = self._formatOntologyTermObject(
                request.age_of_onset, 'phenotype_quality')
            if ontolgytermsClause:
                filters.append(ontolgytermsClause)
        return filters