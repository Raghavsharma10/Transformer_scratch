def _parse_for_errors(self):
        """ Look for an error tag and raise APIError for fatal errors or APIWarning for nonfatal ones. """
        error = self._response.find('{www.clusterpoint.com}error')
        if error is not None:
            if error.find('level').text.lower() in ('rejected', 'failed', 'error', 'fatal'):
                raise APIError(error)
            else:
                warnings.warn(APIWarning(error))