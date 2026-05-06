def __create_phantom_js_driver(self):
        '''
        Creates an instance of PhantomJS driver.
        '''
        try:
            return webdriver.PhantomJS(executable_path=self._config_reader.get(self.PHANTOMEJS_EXEC_PATH),
                                       service_args=['--ignore-ssl-errors=true'])
        except KeyError:
            return webdriver.PhantomJS(service_args=['--ignore-ssl-errors=true'])