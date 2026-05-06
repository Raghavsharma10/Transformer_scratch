def receipt(self, msg):
        """Called to handle a receipt message received from the server.

        This method just logs the receipt message

        returned:
            NO_RESPONSE_NEEDED

        """
        body = msg['body'].replace(NULL, '')

        brief_msg = ""
        if 'receipt-id' in msg['headers']:
            brief_msg = msg['headers']['receipt-id']

        self.log.info("Received server receipt message - receipt-id:%s\n\n%s" % (brief_msg, body))

        returned = NO_RESPONSE_NEEDED
        if self.testing:
            returned = 'receipt'

        return returned