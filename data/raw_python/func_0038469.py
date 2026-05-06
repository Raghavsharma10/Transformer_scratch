def _process_response(response: requests.Response, expected: list = []) -> dict:
        """
        Processes an API response. Raises an exception when appropriate.

        The exception that will be raised is MoneyBird.APIError. This exception is subclassed so implementing programs
        can easily react appropriately to different exceptions.

        The following subclasses of MoneyBird.APIError are likely to be raised:
          - MoneyBird.Unauthorized: No access to the resource or invalid authentication
          - MoneyBird.Throttled: Access (temporarily) denied, please try again
          - MoneyBird.NotFound: Resource not found, check resource path
          - MoneyBird.InvalidData: Validation errors occured while processing your input
          - MoneyBird.ServerError: Error on the server

        :param response: The response to process.
        :param expected: A list of expected status codes which won't raise an exception.
        :return: The useful data in the response (may be None).
        """
        responses = {
            200: None,
            201: None,
            204: None,
            400: MoneyBird.Unauthorized,
            401: MoneyBird.Unauthorized,
            403: MoneyBird.Throttled,
            404: MoneyBird.NotFound,
            406: MoneyBird.NotFound,
            422: MoneyBird.InvalidData,
            429: MoneyBird.Throttled,
            500: MoneyBird.ServerError,
        }

        logger.debug("API request: %s %s\n" % (response.request.method, response.request.url) +
                     "Response: %s %s" % (response.status_code, response.text))

        if response.status_code not in expected:
            if response.status_code not in responses:
                logger.error("API response contained unknown status code")
                raise MoneyBird.APIError(response, "API response contained unknown status code")
            elif responses[response.status_code] is not None:
                try:
                    description = response.json()['error']
                except (AttributeError, TypeError, KeyError, ValueError):
                    description = None
                raise responses[response.status_code](response, description)

        try:
            data = response.json()
        except ValueError:
            logger.error("API response is not JSON decodable")
            data = None

        return data