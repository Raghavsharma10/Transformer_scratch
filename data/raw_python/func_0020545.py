def metadata(self, delete=False):
        """
        Gets the metadata.
        """
        if delete:
            return self._session.delete(self.__v1() + "/metadata").json()
        else:
            return self._session.get(self.__v1() + "/metadata").json()