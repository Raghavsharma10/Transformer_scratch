def get_queryset(self):
        """
        This view should return a list of all the Identities
        for the supplied query parameters. The query parameters
        should be in the form:
        {"address_type": "address"}
        e.g.
        {"msisdn": "+27123"}
        {"email": "foo@bar.com"}

        A special query paramater "include_inactive" can also be passed
        as False to prevent returning identities for which the addresses
        have been set to "inactive"
        e.g.
        {"include_inactive": False}
        """
        query_params = list(self.request.query_params.keys())

        # variable that stores criteria to filter identities by
        filter_criteria = {}
        # variable that stores a list of addresses that should be active
        # if the special filter is passed in
        exclude_if_address_inactive = []

        # Determine from param "include_inactive" whether inactive identities
        # should be included in the search results
        if "include_inactive" in query_params:
            if self.request.query_params["include_inactive"] in [
                "False",
                "false",
                False,
            ]:
                include_inactive = False
            else:
                include_inactive = True
        else:
            include_inactive = True  # default to True

        # Compile a list of criteria to filter the identities by, based on the
        # query parameters
        for filter in query_params:
            if filter in ["include_inactive", "cursor"]:
                # Don't add the cursor to the filter_criteria
                pass
            elif filter.startswith("details__addresses__"):
                # Edit the query_param to evaluate the key instead of the value
                # and add it to the filter_criteria
                filter_criteria[filter + "__has_key"] = self.request.query_params[
                    filter
                ]

                # Add the address to the list of addresses that should not
                # be inactive (tuple e.g ("msisdn", "+27123"))
                if include_inactive is False:
                    exclude_if_address_inactive.append(
                        (
                            filter.replace("details__addresses__", ""),
                            self.request.query_params[filter],
                        )
                    )
            else:
                # Add the normal params to the filter criteria
                filter_criteria[filter] = self.request.query_params[filter]

        identities = Identity.objects.filter(**filter_criteria)

        if include_inactive is False:
            # Check through all the identities and exclude ones where the
            # addresses are inactive
            for identity in identities:
                for param in exclude_if_address_inactive:
                    q_key = identity.details["addresses"][param[0]][param[1]]
                    if "inactive" in q_key and q_key["inactive"] in [
                        True,
                        "True",
                        "true",
                    ]:  # noqa
                        identities = identities.exclude(id=identity.id)

        return identities