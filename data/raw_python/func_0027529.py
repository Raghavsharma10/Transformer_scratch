def xpath(self, xpath, **kwargs):
        """
        Perform an XPath query on the current node.

        :param string xpath: XPath query.
        :param dict kwargs: Optional keyword arguments that are passed through
            to the underlying XML library implementation.

        :return: results of the query as a list of :class:`Node` objects, or
            a list of base type objects if the XPath query does not reference
            node objects.
        """
        result = self.adapter.xpath_on_node(self.impl_node, xpath, **kwargs)
        if isinstance(result, (list, tuple)):
            return [self._maybe_wrap_node(r) for r in result]
        else:
            return self._maybe_wrap_node(result)