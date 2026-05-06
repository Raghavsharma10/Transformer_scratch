def _handle_api_error(self, error):
        """
        New Relic cheerfully provides expected API error codes depending on your
        API call deficiencies so we convert these to exceptions and raise them
        for the user to handle as they see fit.
        """
        status_code = error.response.status_code
        message = error.message

        if 403 == status_code:
            raise NewRelicInvalidApiKeyException(message)
        elif 404 == status_code:
            raise NewRelicUnknownApplicationException(message)
        elif 422 == status_code:
            raise NewRelicInvalidParameterException(message)
        else:
            raise NewRelicApiException(message)