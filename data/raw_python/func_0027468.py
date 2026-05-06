def request(self, method, url, parameters=dict()):
        """Requests wrapper function"""

        # The requests library uses urllib, which serializes to "True"/"False" while Pingdom requires lowercase
        parameters = self._serializeBooleans(parameters)

        headers = {'App-Key': self.apikey}
        if self.accountemail:
            headers.update({'Account-Email': self.accountemail})

        # Method selection handling
        if method.upper() == 'GET':
            response = requests.get(self.url + url, params=parameters,
                                    auth=(self.username, self.password),
                                    headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(self.url + url, data=parameters,
                                     auth=(self.username, self.password),
                                     headers=headers)
        elif method.upper() == 'PUT':
            response = requests.put(self.url + url, data=parameters,
                                    auth=(self.username, self.password),
                                    headers=headers)
        elif method.upper() == 'DELETE':
            response = requests.delete(self.url + url, params=parameters,
                                       auth=(self.username, self.password),
                                       headers=headers)
        else:
            raise Exception("Invalid method in pingdom request")

        # Store pingdom api limits
        self.shortlimit = response.headers.get(
            'Req-Limit-Short',
            self.shortlimit)
        self.longlimit = response.headers.get(
            'Req-Limit-Long',
            self.longlimit)

        # Verify OK response
        if response.status_code != 200:
            sys.stderr.write('ERROR from %s: %d' % (response.url,
                                                    response.status_code))
            sys.stderr.write('Returned data: %s\n' % response.json())
            response.raise_for_status()

        return response