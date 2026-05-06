def request_parking(self, endpoint, url_args={}, **kwargs):
        """Make a request to the given endpoint of the ``parking`` server.

        This returns the plain JSON (dict) response which can then be parsed
        using one of the implemented types.

        Args:
            endpoint (str): Endpoint to send the request to.
                This string corresponds to the key in the ``ENDPOINTS`` dict.
            url_args (dict): Dictionary for URL string replacements.
            **kwargs: Request arguments.

        Returns:
            Obtained response (dict) or None if the endpoint was not found.
        """
        if endpoint not in ENDPOINTS_PARKING:
            # Unknown endpoint
            return None

        url = URL_OPENBUS + ENDPOINTS_PARKING[endpoint]

        # Append additional info to URL
        lang = url_args.get('lang', 'ES')
        address = url_args.get('address', '')

        url = url.format(
            id_client=self._emt_id,
            passkey=self._emt_pass,
            address=address,
            lang=lang
        )

        # This server uses TLSv1
        return _parking_req.post(url, data=kwargs).json()