def view_cookies(self):
        """
        View current cookies in the `requests.Session()` object

        **Returns:** List of Dicts, one cookie per Dict.
        """
        return_list = []
        for cookie in self._session.cookies:
            return_list.append(vars(cookie))

        return return_list