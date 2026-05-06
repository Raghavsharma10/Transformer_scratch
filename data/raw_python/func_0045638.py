def __strip_extra_attributes(self, node: yaml.Node,
                                 known_attrs: List[str]) -> None:
        """Strips tags from extra attributes.

        This prevents nodes under attributes that are not part of our \
        data model from being converted to objects. They'll be plain \
        CommentedMaps instead, which then get converted to OrderedDicts \
        for the user.

        Args:
            node: The node to process
            known_attrs: The attributes to not strip
        """
        known_keys = list(known_attrs)
        known_keys.remove('self')
        if 'yatiml_extra' in known_keys:
            known_keys.remove('yatiml_extra')

        for key_node, value_node in node.value:
            if (not isinstance(key_node, yaml.ScalarNode)
                    or key_node.tag != 'tag:yaml.org,2002:str'):
                raise RecognitionError(
                    ('{}{}Mapping keys that are not of type'
                     ' string are not supported by YAtiML.').format(
                         node.start_mark, os.linesep))
            if key_node.value not in known_keys:
                self.__strip_tags(value_node)