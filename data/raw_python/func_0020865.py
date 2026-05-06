def get_user(self, idvalue, options=None):
        """
        get user by a given id
        http://getstarted.sailthru.com/api/user
        """
        options = options or {}
        data = options.copy()
        data['id'] = idvalue
        return self.api_get('user', data)