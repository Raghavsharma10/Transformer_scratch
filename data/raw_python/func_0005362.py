def resource(
        self,
        token: dict = None,
        id_resource: str = None,
        subresource=None,
        include: list = [],
        prot: str = "https",
    ) -> dict:
        """Get complete or partial metadata about one specific resource.

        :param str token: API auth token
        :param str id_resource: metadata UUID to get
        :param list include: subresources that should be included.
         Must be a list of strings. Available values: 'isogeo.SUBRESOURCES'
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # if subresource route
        if isinstance(subresource, str):
            subresource = "/{}".format(checker._check_subresource(subresource))
        else:
            subresource = ""
            # _includes specific parsing
            include = checker._check_filter_includes(include)

        # handling request parameters
        payload = {"id": id_resource, "_include": include}
        # resource search
        md_url = "{}://v1.{}.isogeo.com/resources/{}{}".format(
            prot, self.api_url, id_resource, subresource
        )
        resource_req = self.get(
            md_url,
            headers=self.header,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )
        checker.check_api_response(resource_req)

        # end of method
        return resource_req.json()