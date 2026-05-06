def xml19139(
        self,
        token: dict = None,
        id_resource: str = None,
        proxy_url=None,
        prot: str = "https",
    ):
        """Get resource exported into XML ISO 19139.

        :param str token: API auth token
        :param str id_resource: metadata UUID to export
        :param str proxy_url: proxy to use to download
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # check metadata UUID
        if not checker.check_is_uuid(id_resource):
            raise ValueError("Metadata ID is not a correct UUID.")
        else:
            pass

        # handling request parameters
        payload = {"proxyUrl": proxy_url, "id": id_resource}

        # resource search
        md_url = "{}://v1.{}.isogeo.com/resources/{}.xml".format(
            prot, self.api_url, id_resource
        )
        xml_req = self.get(
            md_url,
            headers=self.header,
            stream=True,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )

        # end of method
        return xml_req