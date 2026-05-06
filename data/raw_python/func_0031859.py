def build_mongo_doc(self, key_list):
        """
            Creates the components of aggregation pipeline
            :param key_list: list of key which will be used to create the components of aggregation pipeline
            :type  key_list: list

            :returns mongo_doc: dict
        """
        mongo_doc = {}

        if isinstance(key_list, list) and key_list:

            for key in key_list:
                mongo_doc[key] = "$" + str(key)
        return mongo_doc