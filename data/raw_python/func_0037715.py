def add_recipients(self, id, recipients):
        """
        Add recipients.

        Add recipients to an existing group conversation. Response is similar to
        the GET/show action, except that only includes the
        latest message (e.g. "joe was added to the conversation by bob")
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - recipients
        """An array of recipient ids. These may be user ids or course/group ids
        prefixed with "course_" or "group_" respectively, e.g.
        recipients[]=1&recipients[]=2&recipients[]=course_3"""
        data["recipients"] = recipients

        self.logger.debug("POST /api/v1/conversations/{id}/add_recipients with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/conversations/{id}/add_recipients".format(**path), data=data, params=params, no_data=True)