def request_get_user(self, user_ids) -> dict:
        """
        Method to get users by ID, do not need authorization
        """
        method_params = {'user_ids': user_ids}
        response = self.session.send_method_request('users.get', method_params)
        self.check_for_errors('users.get', method_params, response)
        return response