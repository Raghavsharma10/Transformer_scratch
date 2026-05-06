def request_matches_route(self, actual_route: str, expected_route: str):
        """
        Determines whether a route matches the actual requested route or not
        :param actual_route str
        :param expected_route
        :rtype: Boolean
        """
        expected_params = self.get_url_params(expected_route)
        actual_params = self.get_url_params(actual_route)
        i = 0

        if len(expected_params) == len(actual_params):
            for param in actual_params:
                if expected_params[i][0] != "{":
                    if param != expected_params[i]:
                        return False
                i += 1
        else:
            return False

        return True