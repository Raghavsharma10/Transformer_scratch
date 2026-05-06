def _decode_value(self, value):
        """ Decodes the value by turning any binary data back into Python objects.

        The method searches for ObjectId values, loads the associated binary data from
        GridFS and returns the decoded Python object.

        Args:
            value (object): The value that should be decoded.

        Raises:
            DataStoreDecodingError: An ObjectId was found but the id is not a valid
                GridFS id.
            DataStoreDecodeUnknownType: The type of the specified value is unknown.

        Returns:
            object: The decoded value as a valid Python object.
        """
        if isinstance(value, (int, float, str, bool, datetime)):
            return value
        elif isinstance(value, list):
            return [self._decode_value(item) for item in value]
        elif isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result[key] = self._decode_value(item)
            return result
        elif isinstance(value, ObjectId):
            if self._gridfs.exists({"_id": value}):
                return pickle.loads(self._gridfs.get(value).read())
            else:
                raise DataStoreGridfsIdInvalid()
        else:
            raise DataStoreDecodeUnknownType()