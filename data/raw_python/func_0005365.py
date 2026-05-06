def licenses(
        self, token: dict = None, owner_id: str = None, prot: str = "https"
    ) -> dict:
        """Get information about licenses owned by a specific workgroup.

        :param str token: API auth token
        :param str owner_id: workgroup UUID
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # handling request parameters
        payload = {"gid": owner_id}

        # search request
        licenses_url = "{}://v1.{}.isogeo.com/groups/{}/licenses".format(
            prot, self.api_url, owner_id
        )
        licenses_req = self.get(
            licenses_url,
            headers=self.header,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )

        # checking response
        req_check = checker.check_api_response(licenses_req)
        if isinstance(req_check, tuple):
            return req_check

        # end of method
        return licenses_req.json()