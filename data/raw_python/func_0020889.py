def get_last_rate_limit_info(self, action, method):
        """
        Get rate limit information for last API call
        :param action: API endpoint
        :param method: Http method, GET, POST or DELETE
        :return: dict|None
        """
        method = method.upper()
        if (action in self.last_rate_limit_info and method in self.last_rate_limit_info[action]):
            return self.last_rate_limit_info[action][method]

        return None