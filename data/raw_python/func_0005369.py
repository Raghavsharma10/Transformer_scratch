def keywords(
        self,
        token: dict = None,
        thez_id: str = "1616597fbc4348c8b11ef9d59cf594c8",
        query: str = "",
        offset: int = 0,
        order_by: str = "text",
        order_dir: str = "desc",
        page_size: int = 20,
        specific_md: list = [],
        specific_tag: list = [],
        include: list = [],
        prot: str = "https",
    ) -> dict:
        """Search for keywords within a specific thesaurus.

        :param str token: API auth token
        :param str thez_id: thesaurus UUID
        :param str query: search terms
        :param int offset: pagination start
        :param str order_by: sort criteria. Available values :
        
            - count.group,
            - count.isogeo,
            - text

        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # specific resources specific parsing
        specific_md = checker._check_filter_specific_md(specific_md)
        # sub resources specific parsing
        include = checker._check_filter_includes(include, "keyword")
        # specific tag specific parsing
        specific_tag = checker._check_filter_specific_tag(specific_tag)

        # handling request parameters
        payload = {
            "_id": specific_md,
            "_include": include,
            "_limit": page_size,
            "_offset": offset,
            "_tag": specific_tag,
            "tid": thez_id,
            "ob": order_by,
            "od": order_dir,
            "q": query,
        }

        # search request
        keywords_url = "{}://v1.{}.isogeo.com/thesauri/{}/keywords/search".format(
            prot, self.api_url, thez_id
        )

        kwds_req = self.get(
            keywords_url,
            headers=self.header,
            params=payload,
            proxies=self.proxies,
            verify=self.ssl,
        )

        # checking response
        checker.check_api_response(kwds_req)

        # end of method
        return kwds_req.json()