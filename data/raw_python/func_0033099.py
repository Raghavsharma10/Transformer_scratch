def result(self):
        """
        Returns the result from the worker for this job. This is used to pass
        result in async way.
        """
        if self.__result:
            return self.__result
        data = self.rdb.rpop(self.urn)
        if data:
            self.rdb.delete(self.urn)
            data = json.loads(data)
            self.__result = data
            return data
        else:
            return None