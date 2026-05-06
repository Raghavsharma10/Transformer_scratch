def check_auth(self):
        """Perform a re-login if not signed in or raise an exception"""
        if not self.logged_in:
            if self.relogin_interval and self.last_login and (time() - self.last_login) > self.relogin_interval:
                self.log(WARNING, 'Zabbix API not logged in. Performing Zabbix API relogin after %d seconds',
                         self.relogin_interval)
                self.relogin()  # Will raise exception in case of login error
            else:
                raise ZabbixAPIException('Not logged in.')