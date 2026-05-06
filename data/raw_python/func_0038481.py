def delete_grading_period_accounts(self, id, account_id):
        """
        Delete a grading period.

        <b>204 No Content</b> response code is returned if the deletion was
        successful.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("DELETE /api/v1/accounts/{account_id}/grading_periods/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/accounts/{account_id}/grading_periods/{id}".format(**path), data=data, params=params, no_data=True)