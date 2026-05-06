def post(self, url, data, proto='http', form_name=None):
        """
        Load an url using the POST method.

        Keyword arguments:
        url -- the Universal Resource Location
        data -- the form to be sent
        proto -- the protocol (default 'http')
        form_name -- the form name to search the default values
        """
        form = self.translator.fill_form(self.last_response_soup,
                                         form_name if form_name else url, data)
        self.last_response = self.session.post(proto + self.base_uri + url,
                                               headers=self.headers,
                                               cookies=self.cookies,
                                               data=form,
                                               allow_redirects=True,
                                               verify=self.verify)
        return self.last_response_soup