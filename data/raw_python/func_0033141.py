def _type_container(self, value, _type):
        ' apply type to all values in the list '
        if value is None:
            # normalize null containers to empty list
            return []
        elif not isinstance(value, list):
            raise ValueError("expected list type, got: %s" % type(value))
        else:
            return sorted(self._type_single(item, _type) for item in value)