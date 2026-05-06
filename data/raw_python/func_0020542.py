def simulation(self, data=None):
        """
        Gets / Sets the simulation data.

        If no data is passed in, then this method acts as a getter.
        if data is passed in, then this method acts as a setter.

        Keyword arguments:
        data -- the simulation data you wish to set (default None)
        """
        if data:
            return self._session.put(self.__v2() + "/simulation", data=data)
        else:
            return self._session.get(self.__v2() + "/simulation").json()