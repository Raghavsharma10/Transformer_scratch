def iter_json_pages(self, path, page_size=1000, **params):
        """Return an iterator over JSON items from a paginated resource

        Legacy resources (prior to V1) implemented a common paging interfaces for
        several different resources.  This method handles the details of iterating
        over the paged result set, yielding only the JSON data for each item
        within the aggregate resource.

        :param str path: The base path to the resource being requested (e.g. /ws/Group)
        :param int page_size: The number of items that should be requested for each page.  A larger
            page_size may mean fewer HTTP requests but could also increase the time to get a first
            result back from Device Cloud.
        :param params: These are additional query parameters that should be sent with each
            request to Device Cloud.

        """
        path = validate_type(path, *six.string_types)
        page_size = validate_type(page_size, *six.integer_types)

        offset = 0
        remaining_size = 1  # just needs to be non-zero
        while remaining_size > 0:
            reqparams = {"start": offset, "size": page_size}
            reqparams.update(params)
            response = self.get_json(path, params=reqparams)
            offset += page_size
            remaining_size = int(response.get("remainingSize", "0"))
            for item_json in response.get("items", []):
                yield item_json