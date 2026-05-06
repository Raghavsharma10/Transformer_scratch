def _process_output(self, node, **kwargs):
        """
        Processes an output node, which will contain things like `Name` and `TemplateData` nodes.
        """
        for n in node.nodes:
            self._process_node(n, **kwargs)