def lock(self):
        """Lock the Sesame. Return True on success, else False."""
        endpoint = API_SESAME_CONTROL_ENDPOINT.format(self.device_id)
        payload = {'type': 'lock'}
        response = self.account.request('POST', endpoint, payload=payload)
        if response is None:
            return False
        if response.status_code == 200 or response.status_code == 204:
            return True
        return False