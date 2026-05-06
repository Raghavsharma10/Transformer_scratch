def shares(self, token: dict = None, prot: str = "https") -> dict:
        """Get information about shares which feed the application.

        :param str token: API auth token
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # passing auth parameter
        shares_url = "{}://v1.{}.isogeo.com/shares/".format(prot, self.api_url)
        shares_req = self.get(
            shares_url, headers=self.header, proxies=self.proxies, verify=self.ssl
        )

        # checking response
        checker.check_api_response(shares_req)

        # end of method
        return shares_req.json()