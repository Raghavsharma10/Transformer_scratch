def _parse_input_data(self, node):
        """
        Parses inputOutput part camunda modeller extensions.
        Args:
            node: SpiffWorkflow Node object.

        Returns:
            Data dict.
        """
        data = DotDict()
        try:
            for nod in self._get_input_nodes(node):
                data.update(self._parse_input_node(nod))
        except Exception as e:
            log.exception("Error while processing node: %s" % node)
        return data