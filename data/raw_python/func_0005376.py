def get_formats(
        self, token: dict = None, format_code: str = None, prot: str = "https"
    ) -> dict:
        """Get formats.

        :param str token: API auth token
        :param str format_code: code of a specific format
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # if specific format
        if isinstance(format_code, str):
            specific_format = "/{}".format(format_code)
        else:
            specific_format = ""

        # search request
        req_url = "{}://v1.{}.isogeo.com/formats{}".format(
            prot, self.api_url, specific_format
        )

        req = self.get(
            req_url, headers=self.header, proxies=self.proxies, verify=self.ssl
        )

        # checking response
        checker.check_api_response(req)

        # end of method
        return req.json()