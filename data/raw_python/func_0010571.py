def _data_from_dotnotation(self, key, default=None):
        """ Returns the MongoDB data from a key using dot notation.

        Args:
            key (str): The key to the field in the workflow document. Supports MongoDB's
                dot notation for embedded fields.
            default (object): The default value that is returned if the key
                does not exist.

        Returns:
            object: The data for the specified key or the default value.
        """
        if key is None:
            raise KeyError('NoneType is not a valid key!')

        doc = self._collection.find_one({"_id": ObjectId(self._workflow_id)})
        if doc is None:
            return default

        for k in key.split('.'):
            doc = doc[k]

        return doc