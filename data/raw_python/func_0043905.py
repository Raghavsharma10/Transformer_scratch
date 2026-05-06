def do_request(self, http_verb, url, headers, data=None):
        """
        :see :func:`~HttpRequest.do_request`
        """
        if data:
            data = json.dumps(data)
        try:
            response = requests.request(http_verb, url, headers=headers, data=data, verify=config.CACERT_PATH)
        except (requests.exceptions.Timeout, requests.exceptions.TooManyRedirects) as exception:
            self._raise_unrecoverable_error_payplug(exception)
        except requests.exceptions.RequestException as exception:
            self._raise_unrecoverable_error_client(exception)

        return response.content, response.status_code, response.headers