def sendMessage(self, to, message, extra={}):
        """
        If the 'to' parameter is a single entry, we will parse it into a list.
        We will merge default values into the request data and the extra parameters
        provided by the user.
        """
        to = to if isinstance(to, list) else [to]
        to = [str(i) for i in to]

        data = {'to': ','.join(to), 'text': message}
        data = self.merge(data, {'callback': 7, 'mo': 1}, extra)

        try:
            content = self.parseLegacy(self.request('http/sendmsg', data));
        except ClickatellError as e:
            # The error that gets catched here will only be raised if the request was for
            # one number only. We can safely assume we are only dealing with a single response
            # here.
            content = {'error': e.message, 'errorCode': e.code, 'To': to[0]}

        # Force all responses to behave like a list, for consistency
        content = content if isinstance(content, list) else [content]
        result = []

        # Sending messages will also result in a "stable" response. The reason
        # for this is that we can't actually know if the request failed or not...a message
        # that could not be delivered is different from a failed request...so errors are returned
        # per message. In the case of global failures (like authentication) all messages will contain
        # the specific error as part of the response body.
        for index, entry in enumerate(content):
            entry = self.merge({'ID': False, 'To': to[index], 'error': False, 'errorCode': False}, entry)
            result.append({
                'id': entry['ID'],
                'destination': entry['To'],
                'error': entry['error'],
                'errorCode': entry['errorCode']
            });

        return result