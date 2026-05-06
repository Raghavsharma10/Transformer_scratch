def get_health(self, consumers=2, messages=100):
        """
        Returns health information on transport & Redis connections.
        """
        data = {'consumers': consumers, 'messages': messages}

        try:
            self._request('GET', '/health', data=json.dumps(data))
            return True
        except SensuAPIException:
            return False