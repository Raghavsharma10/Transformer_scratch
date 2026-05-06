def __check_no_missing_attributes(self, node: yaml.Node,
                                      mapping: CommentedMap) -> None:
        """Checks that all required attributes are present.

        Also checks that they're of the correct type.

        Args:
            mapping: The mapping with subobjects of this object.

        Raises:
            RecognitionError: if an attribute is missing or the type \
                is incorrect.
        """
        logger.debug('Checking presence of required attributes')
        for name, type_, required in class_subobjects(self.class_):
            if required and name not in mapping:
                raise RecognitionError(('{}{}Missing attribute {} needed for'
                                        ' constructing a {}').format(
                                            node.start_mark, os.linesep, name,
                                            self.class_.__name__))
            if name in mapping and not self.__type_matches(
                    mapping[name], type_):
                raise RecognitionError(('{}{}Attribute {} has incorrect type'
                                        ' {}, expecting a {}').format(
                                            node.start_mark, os.linesep, name,
                                            type(mapping[name]), type_))