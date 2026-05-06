def upload_file_sections(self, user_id, section_id, assignment_id):
        """
        Upload a file.

        Upload a file to a submission.
        
        This API endpoint is the first step in uploading a file to a submission as a student.
        See the {file:file_uploads.html File Upload Documentation} for details on the file upload workflow.
        
        The final step of the file upload workflow will return the attachment data,
        including the new file id. The caller can then POST to submit the
        +online_upload+ assignment with these file ids.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - section_id
        """ID"""
        path["section_id"] = section_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        self.logger.debug("POST /api/v1/sections/{section_id}/assignments/{assignment_id}/submissions/{user_id}/files with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/sections/{section_id}/assignments/{assignment_id}/submissions/{user_id}/files".format(**path), data=data, params=params, no_data=True)