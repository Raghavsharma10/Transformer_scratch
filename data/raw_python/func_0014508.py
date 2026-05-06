def _request(self, method, path, data=None, reestablish_session=True):
        """Perform HTTP request for REST API."""
        if path.startswith("http"):
            url = path  # For cases where URL of different form is needed.
        else:
            url = self._format_path(path)

        headers = {"Content-Type": "application/json"}
        if self._user_agent:
            headers['User-Agent'] = self._user_agent

        body = json.dumps(data).encode("utf-8")
        try:
            response = requests.request(method, url, data=body, headers=headers,
                                        cookies=self._cookies, **self._request_kwargs)
        except requests.exceptions.RequestException as err:
            # error outside scope of HTTP status codes
            # e.g. unable to resolve domain name
            raise PureError(err.message)

        if response.status_code == 200:
            if "application/json" in response.headers.get("Content-Type", ""):
                if response.cookies:
                    self._cookies.update(response.cookies)
                else:
                    self._cookies.clear()
                content = response.json()
                if isinstance(content, list):
                    content = ResponseList(content)
                elif isinstance(content, dict):
                    content = ResponseDict(content)
                content.headers = response.headers
                return content
            raise PureError("Response not in JSON: " + response.text)
        elif response.status_code == 401 and reestablish_session:
            self._start_session()
            return self._request(method, path, data, False)
        elif response.status_code == 450 and self._renegotiate_rest_version:
            # Purity REST API version is incompatible.
            old_version = self._rest_version
            self._rest_version = self._choose_rest_version()
            if old_version == self._rest_version:
                # Got 450 error, but the rest version was supported
                # Something really unexpected happened.
                raise PureHTTPError(self._target, str(self._rest_version), response)
            return self._request(method, path, data, reestablish_session)
        else:
            raise PureHTTPError(self._target, str(self._rest_version), response)