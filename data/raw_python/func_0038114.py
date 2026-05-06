def create_enrollment_term(self, account_id, enrollment_term_end_at=None, enrollment_term_name=None, enrollment_term_sis_term_id=None, enrollment_term_start_at=None):
        """
        Create enrollment term.

        Create a new enrollment term for the specified account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - enrollment_term[name]
        """The name of the term."""
        if enrollment_term_name is not None:
            data["enrollment_term[name]"] = enrollment_term_name

        # OPTIONAL - enrollment_term[start_at]
        """The day/time the term starts.
        Accepts times in ISO 8601 format, e.g. 2015-01-10T18:48:00Z."""
        if enrollment_term_start_at is not None:
            data["enrollment_term[start_at]"] = enrollment_term_start_at

        # OPTIONAL - enrollment_term[end_at]
        """The day/time the term ends.
        Accepts times in ISO 8601 format, e.g. 2015-01-10T18:48:00Z."""
        if enrollment_term_end_at is not None:
            data["enrollment_term[end_at]"] = enrollment_term_end_at

        # OPTIONAL - enrollment_term[sis_term_id]
        """The unique SIS identifier for the term."""
        if enrollment_term_sis_term_id is not None:
            data["enrollment_term[sis_term_id]"] = enrollment_term_sis_term_id

        self.logger.debug("POST /api/v1/accounts/{account_id}/terms with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/terms".format(**path), data=data, params=params, single_item=True)