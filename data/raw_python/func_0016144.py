def login(self, user=None, password=None, save=True):
        """Perform a user.login API request"""
        if user and password:
            if save:
                self.__username = user
                self.__password = password
        elif self.__username and self.__password:
            user = self.__username
            password = self.__password
        else:
            raise ZabbixAPIException('No authentication information available.')

        self.last_login = time()
        # Don't print the raw password
        hashed_pw_string = 'md5(%s)' % md5(password.encode('utf-8')).hexdigest()
        self.debug('Trying to login with %r:%r', user, hashed_pw_string)
        obj = self.json_obj('user.login', params={'user': user, 'password': password}, auth=False)
        self.__auth = self.do_request(obj)