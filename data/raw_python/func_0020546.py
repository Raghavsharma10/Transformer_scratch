def records(self, data=None):
        """
        Gets / Sets records.
        """
        if data:
            return self._session.post(
                self.__v1() + "/records", data=data).json()
        else:
            return self._session.get(self.__v1() + "/records").json()