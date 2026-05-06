def compose(self, to, subject, text):
        """Login required.  Sends POST to send a message to a user.  Returns True or raises :class:`exceptions.UnexpectedResponse` if non-"truthy" value in response.
        
        URL: ``http://www.reddit.com/api/compose/``
        
        :param to: username or :class`things.Account` of user to send to
        :param subject: subject of message
        :param text: message body text
        """
        if isinstance(to, Account):
            to = to.name
        data = dict(to=to, subject=subject, text=text)
        j = self.post('api', 'compose', data=data)
        return assert_truthy(j)