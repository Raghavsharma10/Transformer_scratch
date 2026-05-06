def relogin(self):
        """Perform a re-login"""
        try:
            self.__auth = None  # reset auth before relogin
            self.login()
        except ZabbixAPIException as e:
            self.log(ERROR, 'Zabbix API relogin error (%s)', e)
            self.__auth = None  # logged_in() will always return False
            raise