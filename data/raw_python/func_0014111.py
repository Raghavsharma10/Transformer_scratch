def xpath(self, xpath):
    """ Finds another node by XPath originating at the current node. """
    return [self.get_node_factory().create(node_id)
            for node_id in self._get_xpath_ids(xpath).split(",")
            if node_id]