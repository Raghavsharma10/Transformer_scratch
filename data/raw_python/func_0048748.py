def http_call(self, url=None, **kwargs):
        """
        Call the target URL via HTTP and return the JSON result
        """
        if not url:
            url = self.search_url
        http_func, arg_name = self.get_http_method_arg_name()
        # Build the argument dictionary to pass in the http function
        _kwargs = {
            arg_name: kwargs,
        }
        # The actual HTTP call
        response = http_func(
            url=url.format(**kwargs),
            headers=self.get_http_headers(),
            **_kwargs
        )
        # Error handling
        if response.status_code != 200:
            logger.warning('Invalid Request for `%s`', response.url)
            # Raising a "requests" exception
            response.raise_for_status()
        return response.json()