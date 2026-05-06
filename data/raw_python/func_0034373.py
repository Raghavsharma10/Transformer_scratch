def fetch(self):
        """
        Fetches the query and then tries to wrap the data in the model, joining
        as needed, if applicable.
        """
        returnResults = []

        results = self._query.run()
        for result in results:
            if self._join:
                # Because we can tell the models to ignore certian fields,
                # through the protectedItems blacklist, we can nest models by
                # name and have each one act normal and not accidentally store
                # extra data from other models
                item = self._model.fromRawEntry(**result["left"])
                joined = self._join.fromRawEntry(**result["right"])
                item.protectedItems = self._joinedField
                item[self._joinedField] = joined

            else:
                item = self._model.fromRawEntry(**result)

            returnResults.append(item)

        self._documents = returnResults
        return self._documents