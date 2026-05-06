def share(
        self,
        share_id: str,
        token: dict = None,
        augment: bool = False,
        prot: str = "https",
    ) -> dict:
        """Get information about a specific share and its applications.

        :param str token: API auth token
        :param str share_id: share UUID
        :param bool augment: option to improve API response by adding
         some tags on the fly.
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # passing auth parameter
        share_url = "{}://v1.{}.isogeo.com/shares/{}".format(
            prot, self.api_url, share_id
        )
        share_req = self.get(
            share_url, headers=self.header, proxies=self.proxies, verify=self.ssl
        )

        # checking response
        checker.check_api_response(share_req)

        # enhance share model
        share = share_req.json()
        if augment:
            share = utils.share_extender(
                share, self.search(whole_share=1, share=share_id).get("results")
            )
        else:
            pass

        # end of method
        return share