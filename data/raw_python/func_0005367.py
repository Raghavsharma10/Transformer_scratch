def thesauri(self, token: dict = None, prot: str = "https") -> dict:
        """Get list of available thesauri.

        :param str token: API auth token
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # passing auth parameter
        thez_url = "{}://v1.{}.isogeo.com/thesauri".format(prot, self.api_url)
        thez_req = self.get(
            thez_url, headers=self.header, proxies=self.proxies, verify=self.ssl
        )

        # checking response
        checker.check_api_response(thez_req)

        # end of method
        return thez_req.json()