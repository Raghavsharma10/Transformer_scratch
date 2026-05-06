def start_report(self, report, account_id, _parameters=None):
        """
        Start a Report.

        Generates a report instance for the account.
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

        # OPTIONAL - [parameters]
        """The parameters will vary for each report"""
        if _parameters is not None:
            data["[parameters]"] = _parameters

        self.logger.debug("POST /api/v1/accounts/{account_id}/reports/{report} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/reports/{report}".format(**path), data=data, params=params, single_item=True)