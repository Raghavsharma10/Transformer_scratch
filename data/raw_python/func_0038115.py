def list_enrollment_terms(self, account_id, workflow_state=None):
        """
        List enrollment terms.

        Return all of the terms in the account.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - workflow_state
        """If set, only returns terms that are in the given state.
        Defaults to 'active'."""
        if workflow_state is not None:
            self._validate_enum(workflow_state, ["active", "deleted", "all"])
            params["workflow_state"] = workflow_state

        self.logger.debug("GET /api/v1/accounts/{account_id}/terms with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/terms".format(**path), data=data, params=params, data_key='enrollment_terms', all_pages=True)