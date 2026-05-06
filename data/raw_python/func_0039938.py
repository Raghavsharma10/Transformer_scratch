def request_method(self, method: str,
                       **method_kwargs: Union[str, int]) -> dict:
        """
        Process method request and return json with results

        :param method: str: specifies the method, example: "users.get"
        :param method_kwargs: dict: method parameters,
        example: "users_id=1", "fields='city, contacts'"
        """
        response = self.session.send_method_request(method, method_kwargs)
        self.check_for_errors(method, method_kwargs, response)
        return response