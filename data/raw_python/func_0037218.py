def request(self, method, endpoint, payload=None, timeout=5):
        """Send request to API."""
        url = self.api_url + endpoint
        data = None
        headers = {}

        if payload is not None:
            data = json.dumps(payload)
            headers['Content-Type'] = 'application/json'

        try:
            if self.auth_token is not None:
                headers[API_AUTH_HEADER] = self.auth_token
                response = self.session.request(method, url, data=data,
                                                headers=headers,
                                                timeout=timeout)
                if response.status_code != 401:
                    return response

            _LOGGER.debug("Renewing auth token")
            if not self.login(timeout=timeout):
                return None

            # Retry  request
            headers[API_AUTH_HEADER] = self.auth_token
            return self.session.request(method, url, data=data,
                                        headers=headers,
                                        timeout=timeout)
        except requests.exceptions.ConnectionError:
            _LOGGER.warning("Unable to connect to %s", url)
        except requests.exceptions.Timeout:
            _LOGGER.warning("No response from %s", url)

        return None