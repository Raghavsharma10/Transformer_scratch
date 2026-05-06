def _create_error(self, status_code):
        """
        Construct an error message in jsend format.

        :param int status_code: The status code to translate into an error message
        :return: A dictionary in jsend format with the error and the code
        :rtype: dict
        """

        return jsend.error(message=ComodoCA.status_code[status_code], code=status_code)