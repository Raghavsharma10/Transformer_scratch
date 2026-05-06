def check_api_response(self, response):
        """Check API response and raise exceptions if needed.

        :param requests.models.Response response: request response to check
        """
        # check response
        if response.status_code == 200:
            return True
        elif response.status_code >= 400:
            logging.error(
                "{}: {} - {} - URL: {}".format(
                    response.status_code,
                    response.reason,
                    response.json().get("error"),
                    response.request.url,
                )
            )
            return False, response.status_code