def submit_assignment_courses(self, course_id, assignment_id, submission_submission_type, comment_text_comment=None, submission_body=None, submission_file_ids=None, submission_media_comment_id=None, submission_media_comment_type=None, submission_url=None):
        """
        Submit an assignment.

        Make a submission for an assignment. You must be enrolled as a student in
        the course/section to do this.
        
        All online turn-in submission types are supported in this API. However,
        there are a few things that are not yet supported:
        
        * Files can be submitted based on a file ID of a user or group file. However, there is no API yet for listing the user and group files, or uploading new files via the API. A file upload API is coming soon.
        * Media comments can be submitted, however, there is no API yet for creating a media comment to submit.
        * Integration with Google Docs is not yet supported.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        # OPTIONAL - comment[text_comment]
        """Include a textual comment with the submission."""
        if comment_text_comment is not None:
            data["comment[text_comment]"] = comment_text_comment

        # REQUIRED - submission[submission_type]
        """The type of submission being made. The assignment submission_types must
        include this submission type as an allowed option, or the submission will be rejected with a 400 error.
        
        The submission_type given determines which of the following parameters is
        used. For instance, to submit a URL, submission [submission_type] must be
        set to "online_url", otherwise the submission [url] parameter will be
        ignored."""
        self._validate_enum(submission_submission_type, ["online_text_entry", "online_url", "online_upload", "media_recording", "basic_lti_launch"])
        data["submission[submission_type]"] = submission_submission_type

        # OPTIONAL - submission[body]
        """Submit the assignment as an HTML document snippet. Note this HTML snippet
        will be sanitized using the same ruleset as a submission made from the
        Canvas web UI. The sanitized HTML will be returned in the response as the
        submission body. Requires a submission_type of "online_text_entry"."""
        if submission_body is not None:
            data["submission[body]"] = submission_body

        # OPTIONAL - submission[url]
        """Submit the assignment as a URL. The URL scheme must be "http" or "https",
        no "ftp" or other URL schemes are allowed. If no scheme is given (e.g.
        "www.example.com") then "http" will be assumed. Requires a submission_type
        of "online_url" or "basic_lti_launch"."""
        if submission_url is not None:
            data["submission[url]"] = submission_url

        # OPTIONAL - submission[file_ids]
        """Submit the assignment as a set of one or more previously uploaded files
        residing in the submitting user's files section (or the group's files
        section, for group assignments).
        
        To upload a new file to submit, see the submissions {api:SubmissionsApiController#create_file Upload a file API}.
        
        Requires a submission_type of "online_upload"."""
        if submission_file_ids is not None:
            data["submission[file_ids]"] = submission_file_ids

        # OPTIONAL - submission[media_comment_id]
        """The media comment id to submit. Media comment ids can be submitted via
        this API, however, note that there is not yet an API to generate or list
        existing media comments, so this functionality is currently of limited use.
        
        Requires a submission_type of "media_recording"."""
        if submission_media_comment_id is not None:
            data["submission[media_comment_id]"] = submission_media_comment_id

        # OPTIONAL - submission[media_comment_type]
        """The type of media comment being submitted."""
        if submission_media_comment_type is not None:
            self._validate_enum(submission_media_comment_type, ["audio", "video"])
            data["submission[media_comment_type]"] = submission_media_comment_type

        self.logger.debug("POST /api/v1/courses/{course_id}/assignments/{assignment_id}/submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions".format(**path), data=data, params=params, no_data=True)