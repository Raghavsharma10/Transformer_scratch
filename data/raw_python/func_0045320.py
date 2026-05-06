def __recognize_dict(self, node: yaml.Node,
                         expected_type: Type) -> RecResult:
        """Recognize a node that we expect to be a dict of some kind.

        Args:
            node: The node to recognize.
            expected_type: Dict[str, ...something...]

        Returns:
            expected_type if it was recognized, [] otherwise.
        """
        logger.debug('Recognizing as a dict')
        if not issubclass(generic_type_args(expected_type)[0], str):
            raise RuntimeError(
                'YAtiML only supports dicts with strings as keys')
        if not isinstance(node, yaml.MappingNode):
            message = '{}{}Expected a dict/mapping here'.format(
                node.start_mark, os.linesep)
            return [], message
        value_type = generic_type_args(expected_type)[1]
        for _, value in node.value:
            recognized_value_types, message = self.recognize(value, value_type)
            if len(recognized_value_types) == 0:
                return [], message
            if len(recognized_value_types) > 1:
                return [
                    Dict[str, t]  # type: ignore
                    for t in recognized_value_types
                ], message  # type: ignore

        return [expected_type], ''