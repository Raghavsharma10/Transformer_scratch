def list_user_page_views(self, user_id, end_time=None, start_time=None):
        """
        List user page views.

        Return the user's page view history in json format, similar to the
        available CSV download. Pagination is used as described in API basics
        section. Page views are returned in descending order, newest to oldest.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # OPTIONAL - start_time
        """The beginning of the time range from which you want page views."""
        if start_time is not None:
            params["start_time"] = start_time

        # OPTIONAL - end_time
        """The end of the time range from which you want page views."""
        if end_time is not None:
            params["end_time"] = end_time

        self.logger.debug("GET /api/v1/users/{user_id}/page_views with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/{user_id}/page_views".format(**path), data=data, params=params, all_pages=True)