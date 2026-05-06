def login(self, username, password, load=True):
        """
        Set the authentication data in the object, and if load is True
        (default is True) it also retrieve the ip list and the vm list
        in order to build the internal objects list.
        @param (str) username: username of the cloud
        @param (str) password: password of the cloud
        @param (bool) load: define if pre cache the objects.
        @return: None
        """
        self.auth = Auth(username, password)
        if load is True:
            self.get_ip()
            self.get_servers()