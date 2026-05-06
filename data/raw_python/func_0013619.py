def _formatExternalIdentifiers(self, element, element_type):
        """
        Formats several external identifiers for query
        """
        elementClause = None
        elements = []
        if not issubclass(element.__class__, dict):
            element = protocol.toJsonDict(element)
        if element['externalIdentifiers']:
            for _id in element['externalIdentifiers']:
                elements.append(self._formatExternalIdentifier(
                    _id, element_type))
            elementClause = "({})".format(" || ".join(elements))
        return elementClause