def sesames(self):
        """Return list of Sesames."""
        response = self.request('GET', API_SESAME_LIST_ENDPOINT)
        if response is not None and response.status_code == 200:
            return json.loads(response.text)['sesames']

        _LOGGER.warning("Unable to list Sesames")
        return []