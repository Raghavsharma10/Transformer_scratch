def request(self, action, data={}, headers={}, method='GET'):
        """
        Run the HTTP request against the Clickatell API

        :param str  action:     The API action
        :param dict data:       The request parameters
        :param dict headers:    The request headers (if any)
        :param str  method:     The HTTP method

        :return: The request response
        """
        url = ('https' if self.secure else 'http') + '://' + self.endpoint
        url = url + '/' + action

        # Set the User-Agent
        userAgent = "".join(["ClickatellPython/0.1.2", " ", "Python/", platform.python_version()])
        headers = self.merge({ "User-Agent": userAgent }, headers)

        try:
            func = getattr(requests, method.lower())
        except AttributeError:
            raise Exception('HTTP method ' + method + ' unsupported.')

        resp = func(url, params=data, data=json.dumps(data), headers=headers)

        # Set the coding before unwrapping the text
        resp.encoding = 'utf-8'
        content = resp.text
        return content