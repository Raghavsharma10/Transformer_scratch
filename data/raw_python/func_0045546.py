def require_mapping(self) -> None:
        """Require the node to be a mapping."""
        if not isinstance(self.yaml_node, yaml.MappingNode):
            raise RecognitionError(('{}{}A mapping is required here').format(
                self.yaml_node.start_mark, os.linesep))