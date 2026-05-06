def delete_group_category(self, group_category_id):
        """
        Delete a Group Category.

        Deletes a group category and all groups under it. Protected group
        categories can not be deleted, i.e. "communities" and "student_organized".
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_category_id
        """ID"""
        path["group_category_id"] = group_category_id

        self.logger.debug("DELETE /api/v1/group_categories/{group_category_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/group_categories/{group_category_id}".format(**path), data=data, params=params, no_data=True)