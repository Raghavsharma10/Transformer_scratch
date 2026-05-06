def execute_script(self, string, args=None):
        """
        Execute script passed in to function

        @type string:   str
        @value string:  Script to execute
        @type args:     dict
        @value args:    Dictionary representing command line args

        @rtype:         int
        @rtype:         response code
        """
        result = None

        try:
            result = self.driver_wrapper.driver.execute_script(string, args)
            return result
        except WebDriverException:
            if result is not None:
                message = 'Returned: ' + str(result)
            else:
                message = "No message. Check your Javascript source: {}".format(string)

        raise WebDriverJavascriptException.WebDriverJavascriptException(self.driver_wrapper, message)