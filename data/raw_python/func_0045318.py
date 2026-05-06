def __recognize_scalar(self, node: yaml.Node,
                           expected_type: Type) -> RecResult:
        """Recognize a node that we expect to be a scalar.

        Args:
            node: The node to recognize.
            expected_type: The type it is expected to be.

        Returns:
            A list of recognized types and an error message
        """
        logger.debug('Recognizing as a scalar')
        if (isinstance(node, yaml.ScalarNode)
                and node.tag == scalar_type_to_tag[expected_type]):
            return [expected_type], ''
        message = 'Failed to recognize a {}\n{}\n'.format(
            type_to_desc(expected_type), node.start_mark)
        return [], message