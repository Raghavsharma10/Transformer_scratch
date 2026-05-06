def error(self, msg):
        """Called to handle an error message received from the server.

        This method just logs the error message

        returned:
            NO_RESPONSE_NEEDED

        """
        body = msg['body'].replace(NULL, '')

        brief_msg = ""
        if 'message' in msg['headers']:
            brief_msg = msg['headers']['message']

        self.log.error("Received server error - message%s\n\n%s" % (brief_msg, body))

        returned = NO_RESPONSE_NEEDED
        if self.testing:
            returned = 'error'

        return returned