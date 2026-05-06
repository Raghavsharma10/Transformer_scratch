def match(self, models, results, relation):
        """
        Match the eagerly loaded results to their parents.

        :type models: list
        :type results: Collection
        :type relation:  str
        """
        dictionary = self._build_dictionary(results)

        for model in models:
            key = model.get_key()

            if key in dictionary:
                collection = self._related.new_collection(dictionary[key])

                model.set_relation(relation, collection)

        return models