def assign_unassigned_members(self, group_category_id, sync=None):
        """
        Assign unassigned members.

        Assign all unassigned members as evenly as possible among the existing
        student groups.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_category_id
        """ID"""
        path["group_category_id"] = group_category_id

        # OPTIONAL - sync
        """The assigning is done asynchronously by default. If you would like to
        override this and have the assigning done synchronously, set this value
        to true."""
        if sync is not None:
            data["sync"] = sync

        self.logger.debug("POST /api/v1/group_categories/{group_category_id}/assign_unassigned_members with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/group_categories/{group_category_id}/assign_unassigned_members".format(**path), data=data, params=params, single_item=True)