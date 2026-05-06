def request_set_status(self, text: str) -> dict:
        """
        Method to set user status
        """
        method_params = {'text': text}
        response = self.session.send_method_request('status.set',
                                                    method_params)
        self.check_for_errors('status.set', method_params, response)
        return response