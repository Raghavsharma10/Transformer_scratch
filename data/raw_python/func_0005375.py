def get_coordinate_systems(
        self, token: dict = None, srs_code: str = None, prot: str = "https"
    ) -> dict:
        """Get available coordinate systems in Isogeo API.

        :param str token: API auth token
        :param str srs_code: code of a specific coordinate system
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # if specific format
        if isinstance(srs_code, str):
            specific_srs = "/{}".format(srs_code)
        else:
            specific_srs = ""

        # search request
        req_url = "{}://v1.{}.isogeo.com/coordinate-systems{}".format(
            prot, self.api_url, specific_srs
        )

        req = self.get(
            req_url, headers=self.header, proxies=self.proxies, verify=self.ssl
        )

        # checking response
        checker.check_api_response(req)

        # end of method
        return req.json()