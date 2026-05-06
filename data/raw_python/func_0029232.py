def get_resources(self, collections):
        """ Get resources that correspond to values from :collections:.

        :param collections: Collection names for which resources should be
            gathered
        :type collections: list of str
        :return: Gathered resources
        :rtype: list of Resource instances
        """
        res_map = self.request.registry._model_collections
        resources = [res for res in res_map.values()
                     if res.collection_name in collections]
        resources = [res for res in resources if res]
        return set(resources)