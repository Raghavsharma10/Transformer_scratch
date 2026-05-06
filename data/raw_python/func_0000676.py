def _process_state(self):
        """Process the application state configuration.

        Google Alerts manages the account information and alert data through
        some custom state configuration. Not all values have been completely
        enumerated.
        """
        self._log.debug("Capturing state from the request")
        response = self._session.get(url=self.ALERTS_URL, headers=self.HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")
        for i in soup.findAll('script'):
            if i.text.find('window.STATE') == -1:
                continue
            state = json.loads(i.text[15:-1])
            if state != "":
                self._state = state
                self._log.debug("State value set: %s" % self._state)
        return self._state