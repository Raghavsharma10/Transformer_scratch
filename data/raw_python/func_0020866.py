def save_user(self, idvalue, options=None):
        """
        save user by a given id
        http://getstarted.sailthru.com/api/user
        """
        options = options or {}
        data = options.copy()
        data['id'] = idvalue
        return self.api_post('user', data)