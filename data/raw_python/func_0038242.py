def conclude_deactivate_or_delete_enrollment(self, id, course_id, task=None):
        """
        Conclude, deactivate, or delete an enrollment.

        Conclude, deactivate, or delete an enrollment. If the +task+ argument isn't given, the enrollment
        will be concluded.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - task
        """The action to take on the enrollment.
        When inactive, a user will still appear in the course roster to admins, but be unable to participate.
        ("inactivate" and "deactivate" are equivalent tasks)"""
        if task is not None:
            self._validate_enum(task, ["conclude", "delete", "inactivate", "deactivate"])
            params["task"] = task

        self.logger.debug("DELETE /api/v1/courses/{course_id}/enrollments/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/courses/{course_id}/enrollments/{id}".format(**path), data=data, params=params, single_item=True)