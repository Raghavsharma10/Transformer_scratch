def _formatIds(self, element, element_type):
        """
        Formats a set of identifiers for query
        """
        elementClause = None
        if isinstance(element, collections.Iterable):
            elements = []
            for _id in element:
                elements.append('?{} = <{}> '.format(
                    element_type, _id))
            elementClause = "({})".format(" || ".join(elements))
        return elementClause