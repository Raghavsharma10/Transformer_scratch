def recognize(self, node: yaml.Node, expected_type: Type) -> RecResult:
        """Figure out how to interpret this node.

        This is not quite a type check. This function makes a list of \
        all types that match the expected type and also the node, and \
        returns that list. The goal here is not to test validity, but \
        to determine how to process this node further.

        That said, it will recognize built-in types only in case of \
        an exact match.

        Args:
            node: The YAML node to recognize.
            expected_type: The type we expect this node to be, based \
                    on the context provided by our type definitions.

        Returns:
            A list of matching types.
        """
        logger.debug('Recognizing {} as a {}'.format(node, expected_type))
        recognized_types = None
        if expected_type in [
                str, int, float, bool, bool_union_fix, datetime, None,
                type(None)
        ]:
            recognized_types, message = self.__recognize_scalar(
                node, expected_type)
        elif is_generic_union(expected_type):
            recognized_types, message = self.__recognize_union(
                node, expected_type)
        elif is_generic_list(expected_type):
            recognized_types, message = self.__recognize_list(
                node, expected_type)
        elif is_generic_dict(expected_type):
            recognized_types, message = self.__recognize_dict(
                node, expected_type)
        elif expected_type in self.__registered_classes.values():
            recognized_types, message = self.__recognize_user_classes(
                node, expected_type)

        if recognized_types is None:
            raise RecognitionError(
                ('Could not recognize for type {},'
                 ' is it registered?').format(expected_type))
        logger.debug('Recognized types {} matching {}'.format(
            recognized_types, expected_type))
        return recognized_types, message