def create_static_finding(self, application_id, vulnerability_type, description, severity, parameter=None,
                              file_path=None, native_id=None, column=None, line_text=None, line_number=None):
        """
        Creates a static finding with given properties.
        :param application_id: Application identifier number.
        :param vulnerability_type: Name of CWE vulnerability.
        :param description: General description of the issue.
        :param severity: Severity level from 0-8.
        :param parameter: Request parameter for vulnerability.
        :param file_path: Location of source file.
        :param native_id: Specific identifier for vulnerability.
        :param column: Column number for finding vulnerability source.
        :param line_text: Specific line text to finding vulnerability.
        :param line_number: Specific source line number to finding vulnerability.
        """

        if not parameter and not file_path:
            raise AttributeError('Static findings require either parameter or file_path to be present.')

        params = {
            'isStatic': True,
            'vulnType': vulnerability_type,
            'longDescription': description,
            'severity': severity
        }

        if native_id:
            params['nativeId'] = native_id
        if column:
            params['column'] = column
        if line_text:
            params['lineText'] = line_text
        if line_number:
            params['lineNumber'] = line_number

        return self._request('POST', 'rest/applications/' + str(application_id) + '/addFinding', params)