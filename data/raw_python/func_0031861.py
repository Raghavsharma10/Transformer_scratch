def fetch_and_process_data(self, collection, pipeline):
        """
            Fetches and Processess data from the input collection by aggregating using the pipeline

            :param collection: The collection object for which mongo connection has to be made
            :type  collection: MongoCollection

            :param pipeline: The pipeline using which aggregation will be performed
            :type  pipeline: list of dicts

            :return grouped_docs_dict: dict of property_id,metric_count
        """
        collection_cursor = collection.get_mongo_cursor()
        grouped_docs = list(collection_cursor.aggregate(pipeline))
        grouped_docs_dict = {}

        while grouped_docs:
            doc = grouped_docs.pop()
            keys_list = []

            for group_by_key in self.join_keys:
                keys_list.append(doc["_id"].get(group_by_key, None))
            grouped_docs_dict[tuple(keys_list)] = doc['docs']

        return grouped_docs_dict