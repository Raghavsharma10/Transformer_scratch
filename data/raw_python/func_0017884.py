def login(self, email, password):
        """Login to the flightradar24 session

        The API currently uses flightradar24 as the primary data source. The site provides different levels of data based on user plans.
        For users who have signed up for a plan, this method allows to login with the credentials from flightradar24. The API obtains
        a token that will be passed on all the requests; this obtains the data as per the plan limits.

        Args:
            email (str): The email ID which is used to login to flightradar24
            password (str): The password for the user ID

        Example::

            from pyflightdata import FlightData
            f=FlightData()
            f.login(myemail,mypassword)

        """
        response = FlightData.session.post(
            url=LOGIN_URL,
            data={
                'email': email,
                'password': password,
                'remember': 'true',
                'type': 'web'
            },
            headers={
                'Origin': 'https://www.flightradar24.com',
                'Referer': 'https://www.flightradar24.com',
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0'
            }
        )
        response = self._fr24.json_loads_byteified(
            response.content) if response.status_code == 200 else None
        if response:
            token = response['userData']['subscriptionKey']
            self.AUTH_TOKEN = token