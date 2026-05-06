def set(self, key, value, *, section=DataStoreDocumentSection.Data):
        """ Store a value under the specified key in the given section of the document.

        This method stores a value into the specified section of the workflow data store
        document. Any existing value is overridden. Before storing a value, any linked
        GridFS document under the specified key is deleted.

        Args:
            key (str): The key pointing to the value that should be stored/updated.
                It supports MongoDB's dot notation for nested fields.
            value: The value that should be stored/updated.
            section (DataStoreDocumentSection): The section from which the data should
                be retrieved.

        Returns:
            bool: ``True`` if the value could be set/updated, otherwise ``False``.
        """
        key_notation = '.'.join([section, key])

        try:
            self._delete_gridfs_data(self._data_from_dotnotation(key_notation,
                                                                 default=None))
        except KeyError:
            logger.info('Adding new field {} to the data store'.format(key_notation))

        result = self._collection.update_one(
            {"_id": ObjectId(self._workflow_id)},
            {
                "$set": {
                    key_notation: self._encode_value(value)
                },
                "$currentDate": {"lastModified": True}
            }
        )
        return result.modified_count == 1