def update_courses(self, event, account_id, course_ids):
        """
        Update courses.

        Update multiple courses in an account.  Operates asynchronously; use the {api:ProgressController#show progress endpoint}
        to query the status of an operation.
        
        The action to take on each course.  Must be one of 'offer', 'conclude', 'delete', or 'undelete'.
          * 'offer' makes a course visible to students. This action is also called "publish" on the web site.
          * 'conclude' prevents future enrollments and makes a course read-only for all participants. The course still appears
            in prior-enrollment lists.
          * 'delete' completely removes the course from the web site (including course menus and prior-enrollment lists).
            All enrollments are deleted. Course content may be physically deleted at a future date.
          * 'undelete' attempts to recover a course that has been deleted. (Recovery is not guaranteed; please conclude
            rather than delete a course if there is any possibility the course will be used again.) The recovered course
            will be unpublished. Deleted enrollments will not be recovered.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - course_ids
        """List of ids of courses to update. At most 500 courses may be updated in one call."""
        data["course_ids"] = course_ids

        # REQUIRED - event
        """no description"""
        self._validate_enum(event, ["offer", "conclude", "delete", "undelete"])
        data["event"] = event

        self.logger.debug("PUT /api/v1/accounts/{account_id}/courses with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/accounts/{account_id}/courses".format(**path), data=data, params=params, single_item=True)