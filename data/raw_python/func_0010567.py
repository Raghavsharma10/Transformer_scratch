def get(self, key, default=None, *, section=DataStoreDocumentSection.Data):
        """ Return the field specified by its key from the specified section.

        This method access the specified section of the workflow document and returns the
        value for the given key.

        Args:
            key (str): The key pointing to the value that should be retrieved. It supports
                MongoDB's dot notation for nested fields.
            default: The default value that is returned if the key does not exist.
            section (DataStoreDocumentSection): The section from which the data should
                be retrieved.

        Returns:
            object: The value from the field that the specified key is pointing to. If the
                key does not exist, the default value is returned. If no default value
                is provided and the key does not exist ``None`` is returned.
        """
        key_notation = '.'.join([section, key])
        try:
            return self._decode_value(self._data_from_dotnotation(key_notation, default))
        except KeyError:
            return None