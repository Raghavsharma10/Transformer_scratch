def stopMessage(self, apiMsgId):
        """
        See parent method for documentation
        """
        content =  self.parseLegacy(self.request('http/delmsg', {'apimsgid': apiMsgId}))

        return {
            'id': content['ID'],
            'status': content['Status'],
            'description': self.getStatus(content['Status'])
        }