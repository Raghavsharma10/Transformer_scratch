def get_json_ident(request_headers: dict) -> int:
        """
        Defines whether the JSON response will be indented or not
        :param request_headers: dict
        :return: self
        """
        if 'HTTP_USER_AGENT' in request_headers:
            indent = 2 if re.match("[Mozilla]{7}", request_headers['HTTP_USER_AGENT']) else 0
        else:
            indent = 0

        return indent