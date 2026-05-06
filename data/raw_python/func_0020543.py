def destination(self, name=""):
        """
        Gets / Sets the destination data.
        """
        if name:
            return self._session.put(
                self.__v2() + "/hoverfly/destination",
                data={"destination": name}).json()
        else:
            return self._session.get(
                self.__v2() + "/hoverfly/destination").json()