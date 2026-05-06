def reply(self, text):
        """POSTs reply to message with own message.  Returns posted message.
        
        URL: ``http://www.reddit.com/api/comment/``
        
        :param text: body text of message
        """
        data = {
            'thing_id': self.name,
            'id': '#commentreply_{0}'.format(self.name),
            'text': text,
        }
        j = self._reddit.post('api', 'comment', data=data)
        try:
            return self._reddit._thingify(j['json']['data']['things'][0], path=self._path)
        except Exception:
            raise UnexpectedResponse(j)