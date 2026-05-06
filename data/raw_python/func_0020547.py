def delays(self, delays=[]):
        """
        Gets / Sets the delays. 
        """
        if delays:
            return self._session.put(
                self.__v1() + "/delays", data=json.dumps(delays)).json()
        else:
            return self._session.get(self.__v1() + "/delays").json()