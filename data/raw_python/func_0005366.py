def license(self, license_id: str, token: dict = None, prot: str = "https") -> dict:
        """Get details about a specific license.

        :param str token: API auth token
        :param str license_id: license UUID
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # handling request parameters
        payload = {"lid": license_id}

        # search request
        license_url = "{}://v1.{}.isogeo.com/licenses/{}".format(
            prot, self.api_url, license_id
        )
        license_req = self.get(
            license_url,
            headers=self.header,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )

        # checking response
        checker.check_api_response(license_req)

        # end of method
        return license_req.json()