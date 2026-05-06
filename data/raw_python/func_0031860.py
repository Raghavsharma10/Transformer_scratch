def build_pipeline(self, collection):
        """
            Creates aggregation pipeline for aggregation
            :param collection: Mongo collection for aggregation
            :type  collection: MongoCollection

            :return pipeline: list of dicts
        """
        pipeline = []

        if isinstance(collection.where_dict, dict) and collection.where_dict:
            match_dict = {
                "$match": collection.where_dict
            }
            pipeline.append(match_dict)

        group_keys_dict = self.build_mongo_doc(self.join_keys)
        push_dict = self.build_mongo_doc(collection.select_keys)

        group_by_dict = {
            "$group":
                {
                    "_id": group_keys_dict,
                    "docs": {
                        "$push": push_dict
                    }
                }
        }

        pipeline.append(group_by_dict)

        return pipeline