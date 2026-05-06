def get_directives(self, token: dict = None, prot: str = "https") -> dict:
        """Get environment directives which represent INSPIRE limitations.

        :param str token: API auth token
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # search request
        req_url = "{}://v1.{}.isogeo.com/directives".format(prot, self.api_url)

        req = self.get(
            req_url, headers=self.header, proxies=self.proxies, verify=self.ssl
        )

        # checking response
        checker.check_api_response(req)

        # end of method
        return req.json()