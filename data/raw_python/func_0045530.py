def is_scalar(self, typ: Type = _Any) -> bool:
        """Returns True iff this represents a scalar node.

        If a type is given, checks that the ScalarNode represents this \
        type. Type may be `str`, `int`, `float`, `bool`, or `None`.

        If no type is given, any ScalarNode will return True.
        """
        if isinstance(self.yaml_node, yaml.ScalarNode):
            if typ != _Any and typ in scalar_type_to_tag:
                if typ is None:
                    typ = type(None)
                return self.yaml_node.tag == scalar_type_to_tag[typ]

            if typ is _Any:
                return True
            raise ValueError('Invalid scalar type passed to is_scalar()')
        return False