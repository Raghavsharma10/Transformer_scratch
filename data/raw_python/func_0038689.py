def sendMessage(self, to, message, extra={}):
        """
        If the 'to' parameter is a single entry, we will parse it into a list.
        We will merge default values into the request data and the extra parameters
        provided by the user.
        """
        to = to if isinstance(to, list) else [to]
        to = [str(num) for num in to]
        data = {'to': to, 'text': message}
        data = self.merge(data, {'callback': 7, 'mo': 1}, extra)

        content = self.parseRest(self.request('rest/message', data, {}, 'POST'));
        result = []

        # Messages in the REST response will contain errors on the message entry itself.
        for entry in content['message']:
            entry = self.merge({'apiMessageId': False, 'to': data['to'][0], 'error': False, 'errorCode': False}, entry)
            result.append({
                'id': entry['apiMessageId'].encode('utf-8'),
                'destination': entry['to'].encode('utf-8'),
                'error': entry['error']['description'].encode('utf-8') if entry['error'] != False else False,
                'errorCode': entry['error']['code'].encode('utf-8') if entry['error'] != False else False
            });

        return result