def _request(self, http_method, relative_url='', **kwargs):
        """Does actual HTTP request using requests library."""
        # It could be possible to call api.resource.get('/index')
        # but it would be non-intuitive that the path would resolve
        # to root of domain
        relative_url = self._remove_leading_slash(relative_url)

        # Add default kwargs with possible custom kwargs returned by
        # before_request
        new_kwargs = self.default_kwargs().copy()
        custom_kwargs = self.before_request(
            http_method,
            relative_url,
            kwargs.copy()
        )
        new_kwargs.update(custom_kwargs)

        response = requests.request(
            http_method,
            self._join_url(relative_url),
            **new_kwargs
        )

        return self.after_request(response)