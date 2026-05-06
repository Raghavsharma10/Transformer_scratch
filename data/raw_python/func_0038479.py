def create_subgroup_global(self, id, title, description=None, vendor_guid=None):
        """
        Create a subgroup.

        Creates a new empty subgroup under the outcome group with the given title
        and description.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - title
        """The title of the new outcome group."""
        data["title"] = title

        # OPTIONAL - description
        """The description of the new outcome group."""
        if description is not None:
            data["description"] = description

        # OPTIONAL - vendor_guid
        """A custom GUID for the learning standard"""
        if vendor_guid is not None:
            data["vendor_guid"] = vendor_guid

        self.logger.debug("POST /api/v1/global/outcome_groups/{id}/subgroups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/global/outcome_groups/{id}/subgroups".format(**path), data=data, params=params, single_item=True)