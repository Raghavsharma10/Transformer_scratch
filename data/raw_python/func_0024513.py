def parse_node(self, node):
        """
        Overrides ProcessParser.parse_node
        Parses and attaches the inputOutput tags that created by Camunda Modeller

        Args:
            node: xml task node
        Returns:
             TaskSpec
        """
        spec = super(CamundaProcessParser, self).parse_node(node)
        spec.data = self._parse_input_data(node)
        spec.data['lane_data'] = self._get_lane_properties(node)
        spec.defines = spec.data
        service_class = node.get(full_attr('assignee'))
        if service_class:
            self.parsed_nodes[node.get('id')].service_class = node.get(full_attr('assignee'))
        return spec