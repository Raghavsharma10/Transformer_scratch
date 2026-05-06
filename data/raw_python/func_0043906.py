def do_request(self, http_verb, url, headers, data=None):
        """
        :see :func:`~HttpRequest.do_request`
        """
        if data:
            data = json.dumps(data)
        request = urllib.request.Request(url, data, headers)
        request.get_method = lambda: http_verb

        try:
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as response_:
            response = response_
        except urllib.error.URLError as exception:
            if isinstance(exception.reason, socket.timeout):  # Python 2.6
                self._raise_unrecoverable_error_payplug(exception)
            else:
                self._raise_unrecoverable_error_client(exception)
        except socket.timeout as exception:  # Python 2.7+
            self._raise_unrecoverable_error_payplug(exception)
        except http_client.HTTPException as exception:
            self._raise_unrecoverable_error_client(exception)

        return response.read(), response.code, dict(response.info())