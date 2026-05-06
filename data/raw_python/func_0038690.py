def stopMessage(self, apiMsgId):
        """
        See parent method for documentation
        """
        content =  self.parseRest(self.request('rest/message/' + apiMsgId, {}, {}, 'DELETE'))

        return {
            'id': content['apiMessageId'].encode('utf-8'),
            'status': content['messageStatus'].encode('utf-8'),
            'description': self.getStatus(content['messageStatus'])
        }