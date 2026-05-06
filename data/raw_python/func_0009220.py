def authenticate(self, email=None, password=None):
        """
        Authenticate with LendingClub and preserve the user session for future requests.
        This will raise an exception if the login appears to have failed, otherwise it returns True.

        Since Lending Club doesn't seem to have a login API, the code has to try to decide if the login
        worked or not by looking at the URL redirect and parsing the returned HTML for errors.

        Parameters
        ----------
        email : string
            The email of a user on Lending Club
        password : string
            The user's password, for authentication.

        Returns
        -------
        boolean
            True on success or throws an exception on failure.

        Raises
        ------
        session.AuthenticationError
            If authentication failed
        session.NetworkError
            If a network error occurred
        """

        # Get email and password
        if email is None:
            email = self.email
        else:
            self.email = email

        if password is None:
            password = self.__pass
        else:
            self.__pass = password

        # Get them from the user
        if email is None:
            email = raw_input('Email:')
            self.email = email
        if password is None:
            password = getpass.getpass()
            self.__pass = password

        self.__log('Attempting to authenticate: {0}'.format(self.email))

        # Start session
        self.__session = requests.Session()
        self.__session.headers = {
            'Referer': 'https://www.lendingclub.com/',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_3) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.65 Safari/537.31'
        }

        # Set last request time to now
        self.last_request_time = time.time()

        # Send login request to LC
        payload = {
            'login_email': email,
            'login_password': password
        }
        response = self.post('/account/login.action', data=payload, redirects=False)

        # Get URL redirect URL and save the last part of the path as the endpoint
        response_url = response.url
        if response.status_code == 302:
            response_url = response.headers['location']
        endpoint = response_url.split('/')[-1]

        # Debugging
        self.__log('Status code: {0}'.format(response.status_code))
        self.__log('Redirected to: {0}'.format(response_url))
        self.__log('Cookies: {0}'.format(str(response.cookies.keys())))

        # Show query and data that the server received
        if 'x-echo-query' in response.headers:
            self.__log('Query: {0}'.format(response.headers['x-echo-query']))
        if 'x-echo-data' in response.headers:
            self.__log('Data: {0}'.format(response.headers['x-echo-data']))

        # Parse any errors from the HTML
        soup = BeautifulSoup(response.text, "html5lib")
        errors = soup.find(id='master_error-list')
        if errors:
            errors = errors.text.strip()

            # Remove extra spaces and newlines from error message
            errors = re.sub('\t+', '', errors)
            errors = re.sub('\s*\n+\s*', ' * ', errors)

            if errors == '':
                errors = None

        # Raise error
        if errors is not None:
            raise AuthenticationError(errors)

        # Redirected back to the login page...must be an error
        if endpoint == 'login.action':
            raise AuthenticationError('Unknown! Redirected back to the login page without an error message')

        return True