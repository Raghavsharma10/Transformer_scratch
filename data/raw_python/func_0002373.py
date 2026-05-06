def get(self, *args, **kwargs):
        """Perform a GET request againt the Fred API endpoint."""
        location = args[0]
        params = self._get_keywords(location, kwargs)
        url = self._create_path(*args)
        request = requests.get(url, params=params)
        content = request.content
        self._request = request
        return self._output(content)