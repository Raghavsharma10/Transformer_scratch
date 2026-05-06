def require_attribute(self, attribute: str, typ: Type = _Any) -> None:
        """Require an attribute on the node to exist.

        If `typ` is given, the attribute must have this type.

        Args:
            attribute: The name of the attribute / mapping key.
            typ: The type the attribute must have.
        """
        attr_nodes = [
            value_node for key_node, value_node in self.yaml_node.value
            if key_node.value == attribute
        ]
        if len(attr_nodes) == 0:
            raise RecognitionError(
                ('{}{}Missing required attribute {}').format(
                    self.yaml_node.start_mark, os.linesep, attribute))
        attr_node = attr_nodes[0]

        if typ != _Any:
            recognized_types, message = self.__recognizer.recognize(
                attr_node, cast(Type, typ))
            if len(recognized_types) == 0:
                raise RecognitionError(message)