def mode(self, mode=None):
        """
        Gets / Sets the mode.

        If no mode is provided, then this method acts as a getter.

        Keyword arguments:
        mode -- this should either be 'capture' or 'simulate' (default None)
        """
        if mode:
            logging.debug("SWITCHING TO %s" % mode)
            url = self.__v2() + "/hoverfly/mode"
            logging.debug(url)
            return self._session.put(
                url, data=json.dumps({"mode": mode})).json()["mode"]
        else:
            return self._session.get(
                self.__v2() + "/hoverfly/mode").json()["mode"]