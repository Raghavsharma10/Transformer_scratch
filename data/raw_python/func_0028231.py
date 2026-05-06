def process_json_response(self, response):
        """For a json response, check if there was any error and throw exception.
        Otherwise, create a housecanary.response.Response."""

        response_json = response.json()

        # handle errors
        code_key = "code"
        if code_key in response_json and response_json[code_key] != constants.HTTP_CODE_OK:
            code = response_json[code_key]

            message = response_json
            if "message" in response_json:
                message = response_json["message"]
            elif "code_description" in response_json:
                message = response_json["code_description"]

            if code == constants.HTTP_FORBIDDEN:
                raise housecanary.exceptions.UnauthorizedException(code, message)
            if code == constants.HTTP_TOO_MANY_REQUESTS:
                raise housecanary.exceptions.RateLimitException(code, message, response)
            else:
                raise housecanary.exceptions.RequestException(code, message)

        request_url = response.request.url

        endpoint_name = self._parse_endpoint_name_from_url(request_url)

        return Response.create(endpoint_name, response_json, response)