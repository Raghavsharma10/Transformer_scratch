def invoke(self, formPostEmulator):
        """
        Invoke my callable with input from the browser.

        @param formPostEmulator: a dict of lists of strings in a format like a
            cgi-module form post.
        """
        result = self.fromInputs(formPostEmulator)
        result.addCallback(lambda params: self.callable(**params))
        return result