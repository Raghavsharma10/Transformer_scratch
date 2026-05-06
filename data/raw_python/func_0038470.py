def search_videohub(cls, query, filters=None, status=None, sort=None, size=None, page=None):
        """searches the videohub given a query and applies given filters and other bits

        :see: https://github.com/theonion/videohub/blob/master/docs/search/post.md
        :see: https://github.com/theonion/videohub/blob/master/docs/search/get.md

        :param query: query terms to search by
        :type query: str
        :example query: "brooklyn hipsters"  # although, this is a little redundant...

        :param filters: video field value restrictions
        :type filters: dict
        :default filters: None
        :example filters: {"channel": "onion"} or {"series": "Today NOW"}

        :param status: limit the results to videos that are published, scheduled, draft
        :type status: str
        :default status: None
        :example status: "published" or "draft" or "scheduled"

        :param sort: video field related sorting
        :type sort: dict
        :default sort: None
        :example sort: {"title": "desc"} or {"description": "asc"}

        :param size: the page size (number of results)
        :type size: int
        :default size: None
        :example size": {"size": 20}

        :param page: the page number of the results
        :type page: int
        :default page: None
        :example page: {"page": 2}  # note, you should use `size` in conjunction with `page`

        :return: a dictionary of results and meta information
        :rtype: dict
        """
        # construct url
        url = getattr(settings, "VIDEOHUB_API_SEARCH_URL", cls.DEFAULT_VIDEOHUB_API_SEARCH_URL)
        # construct auth headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": settings.VIDEOHUB_API_TOKEN,
        }
        # construct payload
        payload = {
            "query": query,
        }
        if filters:
            assert isinstance(filters, dict)
            payload["filters"] = filters
        if status:
            assert isinstance(status, six.string_types)
            payload.setdefault("filters", {})
            payload["filters"]["status"] = status
        if sort:
            assert isinstance(sort, dict)
            payload["sort"] = sort
        if size:
            assert isinstance(size, (six.string_types, int))
            payload["size"] = size
        if page:
            assert isinstance(page, (six.string_types, int))
            payload["page"] = page
        # send request
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        # raise if not 200
        if res.status_code != 200:
            res.raise_for_status()
        # parse and return response
        return json.loads(res.content)