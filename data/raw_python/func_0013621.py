def _formatOntologyTerm(self, element, element_type):
        """
        Formats the ontology terms for query
        """
        elementClause = None
        if isinstance(element, dict) and element.get('terms'):
            elements = []
            for _term in element['terms']:
                if _term.get('id'):
                    elements.append('?{} = <{}> '.format(
                        element_type, _term['id']))
                else:
                    elements.append('?{} = <{}> '.format(
                        element_type, self._toNamespaceURL(_term['term'])))
            elementClause = "({})".format(" || ".join(elements))
        return elementClause