def css(self, css):
    """ Finds another node by a CSS selector relative to the current node. """
    return [self.get_node_factory().create(node_id)
            for node_id in self._get_css_ids(css).split(",")
            if node_id]