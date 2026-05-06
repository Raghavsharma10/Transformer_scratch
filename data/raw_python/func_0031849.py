def get_mongo_cursor(self, bulk=False):
        """
            Returns Mongo cursor using the class variables

            :param bulk: bulk writer option
            :type bulk: boolean

            :return: mongo collection for which cursor will be created
            :rtype: mongo colection object
        """
        try:
            if self.host:
                if self.port:
                    client = MongoClient(self.host, self.port)
                else:
                    client = MongoClient(
                        self.host, MongoCollection.DEFAULT_PORT)
            else:

                client = MongoClient(self.mongo_uri)

            db = client[self.db_name]
            cursor = db[self.collection]

            if bulk:
                try:
                    return cursor.initialize_unordered_bulk_op()
                except Exception as e:
                    msg = "Mongo Bulk cursor could not be fetched, Error: {error}".format(
                        error=str(e))
                    raise Exception(msg)

            return cursor

        except Exception as e:
            msg = "Mongo Connection could not be established for Mongo Uri: {mongo_uri}, Database: {db_name}, Collection {col}, Error: {error}".format(
                mongo_uri=self.mongo_uri, db_name=self.db_name, col=self.collection, error=str(e))
            raise Exception(msg)