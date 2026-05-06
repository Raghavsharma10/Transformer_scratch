def do_GET(self, ):
        """Handle GET requests

        If the path is '/', a site which extracts the token will be generated.
        This will redirect the user to the '/sucess' page, which shows
        a success message.

        :returns: None
        :rtype: None
        :raises: None
        """
        urld = {self.extract_site_url: 'extract_token_site.html',
                self.success_site_url: 'success_site.html'}
        site = urld.get(self.path)
        if not site:
            log.debug("Requesting false url on login server.")
            self.send_error(404)
            return
        log.debug('Requesting the login server. Responding with %s.', urld)
        self._set_headers()
        self._write_html(site)