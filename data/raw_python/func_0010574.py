def _delete_gridfs_data(self, data):
        """ Delete all GridFS data that is linked by fields in the specified data.

        Args:
            data: The data that is parsed for MongoDB ObjectIDs. The linked GridFs object
                for any ObjectID is deleted.
        """
        if isinstance(data, ObjectId):
            if self._gridfs.exists({"_id": data}):
                self._gridfs.delete(data)
            else:
                raise DataStoreGridfsIdInvalid()
        elif isinstance(data, list):
            for item in data:
                self._delete_gridfs_data(item)
        elif isinstance(data, dict):
            for key, item in data.items():
                self._delete_gridfs_data(item)