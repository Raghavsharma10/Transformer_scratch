def merge_join_docs(self, keys):
        """
            Merges the final list of docs
            :param left_collection_list: 
            :type  left_collection_list: MongoCollection

            :return join: dict
        """

        join = defaultdict(list)

        for key in keys:
            join[key] = self.generate_join_docs_list(
                self.collections_data['left'].get(key, []), self.collections_data['right'].get(key, []))
        return join