def do_call(self, path, method, body=None, headers=None):
        """
        Send an HTTP request to the REST API.

        :param string path: A URL
        :param string method: The HTTP method (GET, POST, etc.) to use
            in the request.
        :param string body: A string representing any data to be sent in the
            body of the HTTP request.
        :param dictionary headers:
            "{header-name: header-value}" dictionary.

        """
        url = urljoin(self.base_url, path)
        try:
            resp = requests.request(method, url, data=body, headers=headers,
                                    auth=self.auth, timeout=self.timeout)
        except requests.exceptions.Timeout as out:
            raise NetworkError("Timeout while trying to connect to RabbitMQ")
        except requests.exceptions.RequestException as err:
            # All other requests exceptions inherit from RequestException
            raise NetworkError("Error during request %s %s" % (type(err), err))

        try:
            content = resp.json()
        except ValueError as out:
            content = None

        # 'success' HTTP status codes are 200-206
        if resp.status_code < 200 or resp.status_code > 206:
            raise HTTPError(content, resp.status_code, resp.text, path, body)
        else:
            if content:
                return content
            else:
                return None