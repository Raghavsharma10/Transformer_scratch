def abort_all_pending_sis_imports(self, account_id):
        """
        Abort all pending SIS imports.

        Abort already created but not processed or processing SIS imports.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        self.logger.debug("PUT /api/v1/accounts/{account_id}/sis_imports/abort_all_pending with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/accounts/{account_id}/sis_imports/abort_all_pending".format(**path), data=data, params=params)