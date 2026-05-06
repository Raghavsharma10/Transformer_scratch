def login(self):
        """
        Logs into Reddit in order to display a personalised front page.
        """
        data = {'user': self.options['username'], 'passwd':
                self.options['password'], 'api_type': 'json'}
        response = self.client.post('http://www.reddit.com/api/login', data=data)
        self.client.modhash = response.json()['json']['data']['modhash']