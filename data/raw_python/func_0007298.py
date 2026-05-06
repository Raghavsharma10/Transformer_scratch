def create_user(self, username, password, name, email):
        """Create a sub account."""
        method = 'POST'
        endpoint = '/rest/v1/users/{}'.format(self.client.sauce_username)
        body = json.dumps({'username': username, 'password': password,
                           'name': name, 'email': email, })
        return self.client.request(method, endpoint, body)