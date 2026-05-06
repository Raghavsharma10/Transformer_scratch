def update_tab_for_course(self, tab_id, course_id, hidden=None, position=None):
        """
        Update a tab for a course.

        Home and Settings tabs are not manageable, and can't be hidden or moved
        
        Returns a tab object
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - tab_id
        """ID"""
        path["tab_id"] = tab_id

        # OPTIONAL - position
        """The new position of the tab, 1-based"""
        if position is not None:
            data["position"] = position

        # OPTIONAL - hidden
        """no description"""
        if hidden is not None:
            data["hidden"] = hidden

        self.logger.debug("PUT /api/v1/courses/{course_id}/tabs/{tab_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/tabs/{tab_id}".format(**path), data=data, params=params, single_item=True)