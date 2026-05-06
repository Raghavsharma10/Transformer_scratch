def getMessageCharge(self, apiMsgId):
        """
        See parent method for documentation
        """
        content = self.parseRest(self.request('rest/message/' + apiMsgId))

        return {
            'id': apiMsgId,
            'status': content['messageStatus'].encode('utf-8'),
            'description': self.getStatus(content['messageStatus']),
            'charge': float(content['charge'])
        }