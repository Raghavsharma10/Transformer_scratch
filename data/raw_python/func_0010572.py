def _encode_value(self, value):
        """ Encodes the value such that it can be stored into MongoDB.

        Any primitive types are stored directly into MongoDB, while non-primitive types
        are pickled and stored as GridFS objects. The id pointing to a GridFS object
        replaces the original value.

        Args:
            value (object): The object that should be encoded for storing in MongoDB.

        Returns:
            object: The encoded value ready to be stored in MongoDB.
        """
        if isinstance(value, (int, float, str, bool, datetime)):
            return value
        elif isinstance(value, list):
            return [self._encode_value(item) for item in value]
        elif isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result[key] = self._encode_value(item)
            return result
        else:
            return self._gridfs.put(Binary(pickle.dumps(value)),
                                    workflow_id=self._workflow_id)