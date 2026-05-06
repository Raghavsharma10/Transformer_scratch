def __recognize_list(self, node: yaml.Node,
                         expected_type: Type) -> RecResult:
        """Recognize a node that we expect to be a list of some kind.

        Args:
            node: The node to recognize.
            expected_type: List[...something...]

        Returns
            expected_type and the empty string if it was recognized,
                    [] and an error message otherwise.
        """
        logger.debug('Recognizing as a list')
        if not isinstance(node, yaml.SequenceNode):
            message = '{}{}Expected a list here.'.format(
                node.start_mark, os.linesep)
            return [], message
        item_type = generic_type_args(expected_type)[0]
        for item in node.value:
            recognized_types, message = self.recognize(item, item_type)
            if len(recognized_types) == 0:
                return [], message
            if len(recognized_types) > 1:
                recognized_types = [
                    List[t]  # type: ignore
                    for t in recognized_types
                ]
                return recognized_types, message

        return [expected_type], ''