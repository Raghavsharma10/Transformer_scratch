def __recognize_union(self, node: yaml.Node,
                          expected_type: Type) -> RecResult:
        """Recognize a node that we expect to be one of a union of types.

        Args:
            node: The node to recognize.
            expected_type: Union[...something...]

        Returns:
            The specific type that was recognized, multiple, or none.
        """
        logger.debug('Recognizing as a union')
        recognized_types = []
        message = ''
        union_types = generic_type_args(expected_type)
        logger.debug('Union types {}'.format(union_types))
        for possible_type in union_types:
            recognized_type, msg = self.recognize(node, possible_type)
            if len(recognized_type) == 0:
                message += msg
            recognized_types.extend(recognized_type)
        recognized_types = list(set(recognized_types))
        if bool in recognized_types and bool_union_fix in recognized_types:
            recognized_types.remove(bool_union_fix)

        if len(recognized_types) == 0:
            return recognized_types, message
        elif len(recognized_types) > 1:
            message = ('{}{}Could not determine which of the following types'
                       ' this is: {}').format(node.start_mark, os.linesep,
                                              recognized_types)
            return recognized_types, message

        return recognized_types, ''