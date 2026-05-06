def _formatExternalIdentifier(self, element, element_type):
        """
        Formats a single external identifier for query
        """
        if "http" not in element['database']:
            term = "{}:{}".format(element['database'], element['identifier'])
            namespaceTerm = self._toNamespaceURL(term)
        else:
            namespaceTerm = "{}{}".format(
                element['database'], element['identifier'])
        comparison = '?{} = <{}> '.format(element_type, namespaceTerm)
        return comparison