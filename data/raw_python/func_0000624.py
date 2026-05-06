def run_suite(self):
        '''
        Run a phantomjs test suite.

         - ``phantomjs_runner`` is mandatory.
         - Either ``url`` or ``url_name`` needs to be defined.
        '''
        if not self.phantomjs_runner:
            raise JsTestException('phantomjs_runner need to be defined')

        url = self.get_url()

        self.phantomjs(self.phantomjs_runner, url, title=self.title)
        self.cleanup()