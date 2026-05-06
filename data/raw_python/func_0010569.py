def push(self, key, value, *, section=DataStoreDocumentSection.Data):
        """ Appends a value to a list in the specified section of the document.

        Args:
            key (str): The key pointing to the value that should be stored/updated.
                It supports MongoDB's dot notation for nested fields.
            value: The value that should be appended to a list in the data store.
            section (DataStoreDocumentSection): The section from which the data should
                be retrieved.

        Returns:
            bool: ``True`` if the value could be appended, otherwise ``False``.
        """
        key_notation = '.'.join([section, key])
        result = self._collection.update_one(
            {"_id": ObjectId(self._workflow_id)},
            {
                "$push": {
                    key_notation: self._encode_value(value)
                },
                "$currentDate": {"lastModified": True}
            }
        )
        return result.modified_count == 1