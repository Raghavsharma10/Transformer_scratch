def call(self, method, params=None):
        """Check authentication and perform actual API request and relogin if needed"""
        start_time = time()
        self.check_auth()
        self.log(INFO, '[%s-%05d] Calling Zabbix API method "%s"', start_time, self.id, method)
        self.log(DEBUG, '\twith parameters: %s', params)

        try:
            return self.do_request(self.json_obj(method, params=params))
        except ZabbixAPIError as ex:
            if self.relogin_interval and any(i in ex.error['data'] for i in self.LOGIN_ERRORS):
                self.log(WARNING, 'Zabbix API not logged in (%s). Performing Zabbix API relogin', ex)
                self.relogin()  # Will raise exception in case of login error
                return self.do_request(self.json_obj(method, params=params))
            raise  # Re-raise the exception
        finally:
            self.log(INFO, '[%s-%05d] Zabbix API method "%s" finished in %g seconds',
                     start_time, self.id, method, (time() - start_time))