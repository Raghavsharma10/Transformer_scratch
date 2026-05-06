def comment(self, parent, text):
        """Login required.  POSTs a comment in response to ``parent``.  Returns :class:`things.Comment` object.
        
        See https://github.com/reddit/reddit/wiki/API%3A-comment.
        
        URL: ``http://www.reddit.com/api/comment/``
        
        :param parent: full id of thing commenting on
        :param text: comment text
        """
        data = dict(parent=parent, text=text)
        j = self.post('api', 'comment', data=data)
        try:
            return self._thingify(j['json']['data']['things'][0])
        except Exception:
            raise UnexpectedResponse(j)