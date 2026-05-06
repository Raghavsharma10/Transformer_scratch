def bulk_cursor_execute(self, bulk_cursor):
        """
            Executes the bulk_cursor

            :param bulk_cursor: Cursor to perform bulk operations
            :type bulk_cursor: pymongo bulk cursor object

            :returns: pymongo bulk cursor object (for bulk operations)
        """
        try:
            result = bulk_cursor.execute()
        except BulkWriteError as bwe:
            msg = "bulk_cursor_execute: Exception in executing Bulk cursor to mongo with {error}".format(
                error=str(bwe))
            raise Exception(msg)
        except Exception as e:
            msg = "Mongo Bulk cursor could not be fetched, Error: {error}".format(
                error=str(e))
            raise Exception(msg)