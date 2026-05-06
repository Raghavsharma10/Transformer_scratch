def require_sequence(self) -> None:
        """Require the node to be a sequence."""
        if not isinstance(self.yaml_node, yaml.SequenceNode):
            raise RecognitionError(('{}{}A sequence is required here').format(
                self.yaml_node.start_mark, os.linesep))