def remove(self, workflow_id):
        """ Removes a document specified by its id from the data store.

        All associated GridFs documents are deleted as well.

        Args:
            workflow_id (str): The id of the document that represents a workflow run.

        Raises:
            DataStoreNotConnected: If the data store is not connected to the server.
        """
        try:
            db = self._client[self.database]
            fs = GridFSProxy(GridFS(db.unproxied_object))

            for grid_doc in fs.find({"workflow_id": workflow_id},
                                    no_cursor_timeout=True):
                fs.delete(grid_doc._id)

            col = db[WORKFLOW_DATA_COLLECTION_NAME]
            return col.delete_one({"_id": ObjectId(workflow_id)})

        except ConnectionFailure:
            raise DataStoreNotConnected()