def create_manual_finding(self, application_id, vulnerability_type, description, severity, full_url=None,
                              native_id=None, path=None):
        """
        Creates a manual finding with given properties.
        :param application_id: Application identification.
        :param vulnerability_type: Name of CWE vulnerability.
        :param description: General description of the issue.
        :param severity: Severity level from 0-8.
        :param full_url: Absolute URL to the page with the vulnerability.
        :param native_id: Specific identifier for vulnerability.
        :param path: Relative path to vulnerability page.
        """

        params = {
            'isStatic': False,
            'vulnType': vulnerability_type,
            'longDescription': description,
            'severity': severity
        }

        if full_url:
            params['fullUrl'] = full_url
        if native_id:
            params['nativeId'] = native_id
        if path:
            params['path'] = path

        return self._request('POST', 'rest/applications/' + str(application_id) + '/addFinding', params)