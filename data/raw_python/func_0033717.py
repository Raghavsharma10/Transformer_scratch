def newMail(self, data, message):
        """Send a mail to a plugit server"""
        r = self.doQuery('mail', method='POST', postParameters={'response_id': str(data), 'message': str(message)})

        if r.status_code == 200:  # Query ok ?
            data = r.json()

            return data['result'] == 'Ok'

        return False