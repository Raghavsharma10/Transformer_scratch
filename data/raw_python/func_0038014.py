def create_new_grading_standard_accounts(self, title, account_id, grading_scheme_entry_name, grading_scheme_entry_value):
        """
        Create a new grading standard.

        Create a new grading standard
        
        If grading_scheme_entry arguments are omitted, then a default grading scheme
        will be set. The default scheme is as follows:
             "A" : 94,
             "A-" : 90,
             "B+" : 87,
             "B" : 84,
             "B-" : 80,
             "C+" : 77,
             "C" : 74,
             "C-" : 70,
             "D+" : 67,
             "D" : 64,
             "D-" : 61,
             "F" : 0,
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - title
        """The title for the Grading Standard."""
        data["title"] = title

        # REQUIRED - grading_scheme_entry[name]
        """The name for an entry value within a GradingStandard that describes the range of the value
        e.g. A-"""
        data["grading_scheme_entry[name]"] = grading_scheme_entry_name

        # REQUIRED - grading_scheme_entry[value]
        """The value for the name of the entry within a GradingStandard.
        The entry represents the lower bound of the range for the entry.
        This range includes the value up to the next entry in the GradingStandard,
        or 100 if there is no upper bound. The lowest value will have a lower bound range of 0.
        e.g. 93"""
        data["grading_scheme_entry[value]"] = grading_scheme_entry_value

        self.logger.debug("POST /api/v1/accounts/{account_id}/grading_standards with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/grading_standards".format(**path), data=data, params=params, single_item=True)