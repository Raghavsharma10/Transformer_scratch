def index_of_reports(self, report, account_id):
        """
        Index of Reports.

        Shows all reports that have been run for the account of a specific type.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - PATH - report
        """ID"""
        path["report"] = report

        self.logger.debug("GET /api/v1/accounts/{account_id}/reports/{report} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/accounts/{account_id}/reports/{report}".format(**path), data=data, params=params, all_pages=True)