def thesaurus(
        self,
        token: dict = None,
        thez_id: str = "1616597fbc4348c8b11ef9d59cf594c8",
        prot: str = "https",
    ) -> dict:
        """Get a thesaurus.

        :param str token: API auth token
        :param str thez_id: thesaurus UUID
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # handling request parameters
        payload = {"tid": thez_id}

        # passing auth parameter
        thez_url = "{}://v1.{}.isogeo.com/thesauri/{}".format(
            prot, self.api_url, thez_id
        )
        thez_req = self.get(
            thez_url,
            headers=self.header,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )

        # checking response
        checker.check_api_response(thez_req)

        # end of method
        return thez_req.json()