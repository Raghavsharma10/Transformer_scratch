def _make_request(self, request, uri, **kwargs):
        """
        This is final step of calling out to the remote API.  We set up our
        headers, proxy, debugging etc. The only things in **kwargs are
        parameters that are overridden on an API method basis (like timeout)

        We have a simple attempt/while loop to implement retries w/ delays to
        avoid the brittleness of talking over the network to remote services.
        These settings can be overridden when creating the Client.

        We catch the 'requests' exceptions during our retries and eventually
        raise our own NewRelicApiException if we are unsuccessful in contacting
        the New Relic API. Additionally we process any non 200 HTTP Errors
        and raise an appropriate exception according to the New Relic API
        documentation.

        Finally we pass back the response text to our XML parser since we have
        no business parsing that here. It could be argued that handling API
        exceptions/errors shouldn't belong in this method but it is simple
        enough for now.
        """
        attempts = 0
        response = None
        while attempts <= self.retries:
            try:
                response = request(uri, headers=self.headers, proxies=self.proxy, **kwargs)

            except (requests.ConnectionError, requests.HTTPError) as ce:
                attempts += 1
                msg = "Attempting retry {attempts} after {delay} seconds".format(attempts=attempts, delay=self.retry_delay)
                logger.error(ce.__doc__)
                logger.error(msg)
                sleep(self.retry_delay)
            else:
                break
        if response is not None:
            try:
                response.raise_for_status()
            except Exception as e:
                self._handle_api_error(e)
        else:
            raise requests.RequestException
        return self._parser(response.text)