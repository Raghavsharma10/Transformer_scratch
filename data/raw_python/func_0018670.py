def _key(self):
        """ Generates the Key object based on dimension fields. """
        return Key(self._schema.key_type, self._identity, self._name,
                   [str(item.value) for item in self._dimension_fields.values()])