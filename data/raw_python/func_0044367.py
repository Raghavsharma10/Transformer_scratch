def get_vulnerabilities(self, teams=None, applications=None, channel_types=None, start_date=None, end_date=None,
                            generic_severities=None, generic_vulnerabilities=None, number_merged=None,
                            number_vulnerabilities=None, parameter=None, path=None, show_open=None, show_closed=None,
                            show_defect_open=None, show_defect_closed=None, show_defect_present=None,
                            show_defect_not_present=None, show_false_positive=None, show_hidden=None):
        """
        Returns filtered list of vulnerabilities.
        :param teams: List of team ids.
        :param applications: List of application ids.
        :param channel_types: List of scanner names.
        :param start_date: Lower bound on scan dates.
        :param end_date: Upper bound on scan dates.
        :param generic_severities: List of generic severity values.
        :param generic_vulnerabilities: List of generic vulnerability ids.
        :param number_merged: Number of vulnerabilities merged from different scans.
        :param number_vulnerabilities: Number of vulnerabilities to return.
        :param parameter: Application input that the vulnerability affects.
        :param path: Path to the web page where the vulnerability was found.
        :param show_open: Flag to show all open vulnerabilities.
        :param show_closed: Flag to show all closed vulnerabilities.
        :param show_defect_open: Flag to show any vulnerabilities with open defects.
        :param show_defect_closed: Flag to show any vulnerabilities with closed defects.
        :param show_defect_present: Flag to show any vulnerabilities with a defect.
        :param show_defect_not_present: Flag to show any vulnerabilities without a defect.
        :param show_false_positive: Flag to show any false positives from vulnerabilities.
        :param show_hidden: Flag to show all hidden vulnerabilities.
        """
        params = {}

        # Build parameter list
        if teams:
            params.update(self._build_list_params('teams', 'id', teams))
        if applications:
            params.update(self._build_list_params('applications', 'id', applications))
        if channel_types:
            params.update(self._build_list_params('channelTypes', 'name', channel_types))
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date
        if generic_severities:
            params.update(self._build_list_params('genericSeverities', 'intValue', generic_severities))
        if generic_vulnerabilities:
            params.update(self._build_list_params('genericVulnerabilities', 'id', generic_vulnerabilities))
        if number_merged:
            params['numberMerged'] = number_merged
        if number_vulnerabilities:
            params['numberVulnerabilities'] = number_vulnerabilities
        if parameter:
            params['parameter'] = parameter
        if path:
            params['path'] = path
        if show_open:
            params['showOpen'] = show_open
        if show_closed:
            params['showClosed'] = show_closed
        if show_defect_open:
            params['showDefectOpen'] = show_defect_open
        if show_defect_closed:
            params['showDefectClosed'] = show_defect_closed
        if show_defect_present:
            params['showDefectPresent'] = show_defect_present
        if show_defect_not_present:
            params['showDefectNotPresent'] = show_defect_not_present
        if show_false_positive:
            params['showFalsePositive'] = show_false_positive
        if show_hidden:
            params['showHidden'] = show_hidden

        return self._request('POST', 'rest/vulnerabilities', params)