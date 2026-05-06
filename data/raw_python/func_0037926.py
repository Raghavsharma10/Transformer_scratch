def get_public_inline_preview_url(self, id, submission_id=None):
        """
        Get public inline preview url.

        Determine the URL that should be used for inline preview of the file.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - submission_id
        """The id of the submission the file is associated with.  Provide this argument to gain access to a file
        that has been submitted to an assignment (Canvas will verify that the file belongs to the submission
        and the calling user has rights to view the submission)."""
        if submission_id is not None:
            params["submission_id"] = submission_id

        self.logger.debug("GET /api/v1/files/{id}/public_url with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/files/{id}/public_url".format(**path), data=data, params=params, no_data=True)