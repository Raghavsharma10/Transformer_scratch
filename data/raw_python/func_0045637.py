def __type_check_attributes(self, node: yaml.Node, mapping: CommentedMap,
                                argspec: inspect.FullArgSpec) -> None:
        """Ensure all attributes have a matching constructor argument.

        This checks that there is a constructor argument with a \
        matching type for each existing attribute.

        If the class has a yatiml_extra attribute, then extra \
        attributes are okay and no error will be raised if they exist.

        Args:
            node: The node we're processing
            mapping: The mapping with constructed subobjects
            constructor_attrs: The attributes of the constructor, \
                    including self and yatiml_extra, if applicable
        """
        logger.debug('Checking for extraneous attributes')
        logger.debug('Constructor arguments: {}, mapping: {}'.format(
            argspec.args, list(mapping.keys())))
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise RecognitionError(('{}{}YAtiML only supports strings'
                                        ' for mapping keys').format(
                                            node.start_mark, os.linesep))
            if key not in argspec.args and 'yatiml_extra' not in argspec.args:
                raise RecognitionError(
                    ('{}{}Found additional attributes'
                     ' and {} does not support those').format(
                         node.start_mark, os.linesep, self.class_.__name__))

            if key in argspec.args and not self.__type_matches(
                    value, argspec.annotations[key]):
                raise RecognitionError(('{}{}Expected attribute {} to be of'
                                        ' type {} but it is a(n) {}').format(
                                            node.start_mark, os.linesep, key,
                                            argspec.annotations[key],
                                            type(value)))