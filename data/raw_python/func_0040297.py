def _is_type(self, instance, type):
        """
        Check if an ``instance`` is of the provided (JSON Schema) ``type``.
        """
        if type not in self._types:
            raise UnknownType(type)
        type = self._types[type]

        # bool inherits from int, so ensure bools aren't reported as integers
        if isinstance(instance, bool):
            type = _flatten(type)
            if int in type and bool not in type:
                return False
        return isinstance(instance, type)