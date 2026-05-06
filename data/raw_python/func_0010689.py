def _check_html_response(self, response):
        """
            Checks if the API Key is valid and if the request returned a 200 status (ok)
        """

        error1 = "Access to this form requires a valid API key. For more info see: http://www.clublog.org/need_api.php"
        error2 = "Invalid or missing API Key"

        if response.status_code == requests.codes.ok:
            return True
        else:
            err_str = "HTTP Status Code: " + str(response.status_code) + " HTTP Response: " + str(response.text)
            self._logger.error(err_str)
            if response.status_code == 403:
                raise APIKeyMissingError
            else:
                raise LookupError(err_str)