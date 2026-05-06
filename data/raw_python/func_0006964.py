def list(self, filters=None):
        """List model instances.

        Currently this gets *everything* and iterates through all
        possible pages in the API. This may be unsuitable for production
        environments with huge databases, so finer grained page support
        should likely be added at some point.

        Args:
            filters (dict, optional): API query filters to apply to the
                request. For example:

                .. code-block:: python

                    {'name__startswith': 'azure',
                     'user__in': [1, 2, 3, 4],}

                See saltant's API reference at
                https://saltant-org.github.io/saltant/ for each model's
                available filters.

        Returns:
            list:
                A list of :class:`saltant.models.resource.Model`
                subclass instances (for example, container task type
                model instances).
        """
        # Add in the page and page_size parameters to the filter, such
        # that our request gets *all* objects in the list. However,
        # don't do this if the user has explicitly included these
        # parameters in the filter.
        if not filters:
            filters = {}

        if "page" not in filters:
            filters["page"] = 1

        if "page_size" not in filters:
            # The below "magic number" is 2^63 - 1, which is the largest
            # number you can hold in a 64 bit integer. The main point
            # here is that we want to get everything in one page (unless
            # otherwise specified, of course).
            filters["page_size"] = 9223372036854775807

        # Form the request URL - first add in the query filters
        query_filter_sub_url = ""

        for idx, filter_param in enumerate(filters):
            # Prepend '?' or '&'
            if idx == 0:
                query_filter_sub_url += "?"
            else:
                query_filter_sub_url += "&"

            # Add in the query filter
            query_filter_sub_url += "{param}={val}".format(
                param=filter_param, val=filters[filter_param]
            )

        # Stitch together all sub-urls
        request_url = (
            self._client.base_api_url + self.list_url + query_filter_sub_url
        )

        # Make the request
        response = self._client.session.get(request_url)

        # Validate that the request was successful
        self.validate_request_success(
            response_text=response.text,
            request_url=request_url,
            status_code=response.status_code,
            expected_status_code=HTTP_200_OK,
        )

        # Return a list of model instances
        return self.response_data_to_model_instances_list(response.json())