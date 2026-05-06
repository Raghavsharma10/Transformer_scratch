def read_message(self, id_):
        """Login required.  Send POST to mark a message as read.  Returns True or raises :class:`exceptions.UnexpectedResponse` if non-"truthy" value in response.
        
        URL: ``http://www.reddit.com/api/read_message/``
        
        :param id\_: full id of message to mark
        """
        data = dict(id=id_)
        j = self.post('api', 'read_message', data=data)
        return assert_truthy(j)