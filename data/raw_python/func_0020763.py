def getHelp(self, call=""):
        """
        API to get a list of supported REST APIs. In the case a particular API is specified,
        the docstring of that API is displayed.

        :param call: call to get detailed information about (Optional)
        :type call: str
        :return: List of APIs or detailed information about a specific call (parameters and docstring)
        :rtype: List of strings or a dictionary containing params and doc keys depending on the input parameter

        """
        if call:
            params = self.methods['GET'][call]['args']
            doc = self.methods['GET'][call]['call'].__doc__
            return dict(params=params, doc=doc)
        else:
            return self.methods['GET'].keys()