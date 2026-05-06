def update_state(self, cache=True):
        """Update the internal state of the Sesame."""
        self.use_cached_state = cache

        endpoint = API_SESAME_ENDPOINT.format(self._device_id)
        response = self.account.request('GET', endpoint)
        if response is None or response.status_code != 200:
            return

        state = json.loads(response.text)
        self._nickname = state['nickname']
        self._is_unlocked = state['is_unlocked']
        self._api_enabled = state['api_enabled']
        self._battery = state['battery']