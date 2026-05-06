def url(self):
        '''Current or base URL. Can be redefined via keyword argument on
        initialization.

        Returns `iktomi.web.URL object.
        `'''
        return URL.from_url(self.request.url, show_host=self.show_host)