def request(self, url, method="get", format="json", data=None,
                expected_status=None, headers=None, use_xpost=True, **options):
        """
        Make an HTTP request to the given relative URL with the host,
        user, and password information. Returns the deserialized json
        if successful, and raises an exception otherwise
        """
        if expected_status is None:
            if method == "get":
                expected_status = 200
            elif method == "post":
                expected_status = 201
            else:
                raise ValueError("No expected status supplied and method unknown.")

        if not url.startswith("http"):
            url = "{self.host}/api/v4/{url}".format(**locals())
        if format is not None:
            options = dict({'format': format}, **options)
        options = {field: value for field, value in options.items() if value is not None}
        headers = dict(headers or {}, Authorization="Token {}".format(self.token))
        #headers['Accept-encoding'] = 'gzip'

        if method == "get" and use_xpost:
            # If method is purely GET, we can use X-HTTP-METHOD-OVERRIDE to send our
            # query via POST. This allows for a large number of parameters to be supplied
            assert(data is None)

            headers.update({"X-HTTP-METHOD-OVERRIDE": method})
            data = options
            options = None
            method = "post"

        r = requests.request(method, url, data=data, params=options, headers=headers)

        log.debug(
            "HTTP {method} {url} (options={options!r}, data={data!r},"
            "headers={headers}) -> {r.status_code}".format(**locals())
        )
        return check(r, expected_status=expected_status)