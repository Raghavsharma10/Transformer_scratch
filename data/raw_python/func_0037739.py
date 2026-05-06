def get_sis_import_list(self, account_id, created_since=None):
        """
        Get SIS import list.

        Returns the list of SIS imports for an account

        Example:
          curl 'https://<canvas>/api/v1/accounts/<account_id>/sis_imports' \
            -H "Authorization: Bearer <token>"
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - created_since
        """If set, only shows imports created after the specified date (use ISO8601 format)"""
        if created_since is not None:
            if issubclass(created_since.__class__, basestring):
                created_since = self._validate_iso8601_string(created_since)
            elif issubclass(created_since.__class__, date) or issubclass(created_since.__class__, datetime):
                created_since = created_since.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            params["created_since"] = created_since

        self.logger.debug("GET /api/v1/accounts/{account_id}/sis_imports with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/sis_imports".format(**path), data=data, params=params, data_key='sis_imports', all_pages=True)