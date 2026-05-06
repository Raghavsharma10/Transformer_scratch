def extend(self, key, values, *, section=DataStoreDocumentSection.Data):
        """ Extends a list in the data store with the elements of values.

        Args:
            key (str): The key pointing to the value that should be stored/updated.
                It supports MongoDB's dot notation for nested fields.
            values (list): A list of the values that should be used to extend the list
                in the document.
            section (DataStoreDocumentSection): The section from which the data should
                be retrieved.

        Returns:
            bool: ``True`` if the list in the database could be extended,
                otherwise ``False``.
        """
        key_notation = '.'.join([section, key])
        if not isinstance(values, list):
            return False

        result = self._collection.update_one(
            {"_id": ObjectId(self._workflow_id)},
            {
                "$push": {
                    key_notation: {"$each": self._encode_value(values)}
                },
                "$currentDate": {"lastModified": True}
            }
        )
        return result.modified_count == 1