def search(
        self,
        token: dict = None,
        query: str = "",
        bbox: list = None,
        poly: str = None,
        georel: str = None,
        order_by: str = "_created",
        order_dir: str = "desc",
        page_size: int = 100,
        offset: int = 0,
        share: str = None,
        specific_md: list = [],
        include: list = [],
        whole_share: bool = True,
        check: bool = True,
        augment: bool = False,
        tags_as_dicts: bool = False,
        prot: str = "https",
    ) -> dict:
        """Search within the resources shared to the application.

        It's the main method to use.

        :param str token: API auth token - DEPRECATED: token is now automatically included
        :param str query: search terms and semantic filters. Equivalent of
         **q** parameter in Isogeo API. It could be a simple
         string like *oil* or a tag like *keyword:isogeo:formations*
         or *keyword:inspire-theme:landcover*. The *AND* operator
         is applied when various tags are passed.
        :param list bbox: Bounding box to limit the search.
         Must be a 4 list of coordinates in WGS84 (EPSG 4326).
         Could be associated with *georel*.
        :param str poly: Geographic criteria for the search, in WKT format.
         Could be associated with *georel*.
        :param str georel: geometric operator to apply to the bbox or poly
         parameters.

         Available values (see: *isogeo.GEORELATIONS*):

          * 'contains',
          * 'disjoint',
          * 'equals',
          * 'intersects' - [APPLIED BY API if NOT SPECIFIED]
          * 'overlaps',
          * 'within'.

        :param str order_by: sorting results.

         Available values:

          * '_created': metadata creation date [DEFAULT if relevance is null]
          * '_modified': metadata last update
          * 'title': metadata title
          * 'created': data creation date (possibly None)
          * 'modified': data last update date
          * 'relevance': relevance score calculated by API [DEFAULT].

        :param str order_dir: sorting direction.

         Available values:
          * 'desc': descending
          * 'asc': ascending

        :param int page_size: limits the number of results.
         Useful to paginate results display. Default value: 100.
        :param int offset: offset to start page size
         from a specific results index
        :param str share: share UUID to filter on
        :param list specific_md: list of metadata UUIDs to filter on
        :param list include: subresources that should be returned.
         Must be a list of strings. Available values: *isogeo.SUBRESOURCES*
        :param bool whole_share: option to return all results or only the
         page size. *True* by DEFAULT.
        :param bool check: option to check query parameters and avoid erros.
         *True* by DEFAULT.
        :param bool augment: option to improve API response by adding
         some tags on the fly (like shares_id)
        :param bool tags_as_dicts: option to store tags as key/values by filter.
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # specific resources specific parsing
        specific_md = checker._check_filter_specific_md(specific_md)

        # sub resources specific parsing
        include = checker._check_filter_includes(include)

        # handling request parameters
        payload = {
            "_id": specific_md,
            "_include": include,
            "_lang": self.lang,
            "_limit": page_size,
            "_offset": offset,
            "box": bbox,
            "geo": poly,
            "rel": georel,
            "ob": order_by,
            "od": order_dir,
            "q": query,
            "s": share,
        }

        if check:
            checker.check_request_parameters(payload)
        else:
            pass

        # search request
        search_url = "{}://v1.{}.isogeo.com/resources/search".format(prot, self.api_url)
        try:
            search_req = self.get(
                search_url,
                headers=self.header,
                params=payload,
                proxies=self.proxies,
                verify=self.ssl,
            )
        except Exception as e:
            logging.error(e)
            raise Exception

        # fast response check
        checker.check_api_response(search_req)

        # serializing result into dict and storing resources in variables
        search_rez = search_req.json()
        resources_count = search_rez.get("total")  # total of metadatas shared

        # handling Isogeo API pagination
        # see: http://help.isogeo.com/api/fr/methods/pagination.html
        if resources_count > page_size and whole_share:
            # if API returned more than one page of results, let's get the rest!
            metadatas = []  # a recipient list
            payload["_limit"] = 100  # now it'll get pages of 100 resources
            # let's parse pages
            for idx in range(0, int(ceil(resources_count / 100)) + 1):
                payload["_offset"] = idx * 100
                search_req = self.get(
                    search_url,
                    headers=self.header,
                    params=payload,
                    proxies=self.proxies,
                    verify=self.ssl,
                )
                # storing results by addition
                metadatas.extend(search_req.json().get("results"))
            search_rez["results"] = metadatas
        else:
            pass

        # add shares to tags and query
        if augment:
            self.add_tags_shares(search_rez.get("tags"))
            if share:
                search_rez.get("query")["_shares"] = [share]
            else:
                search_rez.get("query")["_shares"] = []
        else:
            pass

        # store tags in dicts
        if tags_as_dicts:
            new_tags = utils.tags_to_dict(
                tags=search_rez.get("tags"), prev_query=search_rez.get("query")
            )
            # clear
            search_rez.get("tags").clear()
            search_rez.get("query").clear()
            # update
            search_rez.get("tags").update(new_tags[0])
            search_rez.get("query").update(new_tags[1])
        else:
            pass
        # end of method
        return search_rez