def open(self, page, parms=None, payload=None, HTTPrequest=None ):
        '''Opens a page from the server with optional content.  Returns the string response.'''
        response = self.open_raw( page, parms, payload, HTTPrequest )
        return response.read()