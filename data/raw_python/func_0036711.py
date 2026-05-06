def transfer_data_from_mongo(self,
                                 index,
                                 doc_type,
                                 use_mongo_id=False,
                                 indexed_flag_field_name='',
                                 mongo_query_params={},
                                 mongo_host=default.MONGO_HOST,
                                 mongo_port=default.MONGO_PORT,
                                 mongo_db=default.MONGO_DB,
                                 mongo_collection=default.MONGO_COLLECTION):
        """
        Transfer data from MongoDB into the Elasticsearch, the hostname, port, database and
        collection name in MongoDB default from load in default.py

        :param index: The name of the index
        :param doc_type: The type of the document
        :param use_mongo_id: Use id of MongoDB in the Elasticsearch if is true otherwise automatic generation
        :param indexed_flag_field_name: the name of the field of the document,
                    if associated value is False will synchronize data for it
        :param mongo_client_params: The dictionary for client params of MongoDB
        :param mongo_query_params: The dictionary for query params of MongoDB
        :param mongo_host: The name of the hostname from MongoDB
        :param mongo_port: The number of the port from MongoDB
        :param mongo_db: The name of the database from MongoDB
        :param mongo_collection: The name of the collection from MongoDB
        :return: void
        """
        mongo_client = MongoClient(host=mongo_host, port=int(mongo_port))
        try:
            collection = mongo_client[mongo_db][mongo_collection]
            if indexed_flag_field_name != '':
                mongo_query_params.update({indexed_flag_field_name: False})
            mongo_docs = collection.find(mongo_query_params)
        finally:
            mongo_client.close()
        # Joint actions of Elasticsearch for execute bulk api
        actions = []
        id_array = []
        for doc in mongo_docs:
            action = {
                '_op_type': 'index',
                '_index': index,
                '_type': doc_type
            }
            id_array.append(doc['_id'])
            if not use_mongo_id:
                doc.pop('_id')
            else:
                doc['id'] = str(doc['_id'])
                doc.pop('_id')
            action['_source'] = doc
            actions.append(action)
        success, failed = es_helpers.bulk(self.client, actions, request_timeout=60 * 60)
        logger.info(
            'Transfer data from MongoDB(%s:%s) into the Elasticsearch(%s) success: %s, failed: %s' % (
                mongo_host, mongo_port, self.client, success, failed))

        # Back update flag
        if indexed_flag_field_name != '':
            t = threading.Thread(target=ElasticsearchClient._back_update_mongo,
                                 args=(self, mongo_host, mongo_port, mongo_db, mongo_collection, id_array,
                                       {indexed_flag_field_name: True}),
                                 name='mongodb_back_update')
            t.start()
        return success, failed