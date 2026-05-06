def edit(self, id_, text):
        """Login required.  Sends POST to change selftext or comment text to ``text``.  Returns :class:`things.Comment` or :class:`things.Link` object depending on what's being edited.  Raises :class:`UnexpectedResponse` if neither is returned.
        
        URL: ``http://www.reddit.com/api/editusertext/``
        
        :param id\_: full id of link or comment to edit
        :param text: new self or comment text
        """
        data = dict(thing_id=id_, text=text)
        j = self.post('api', 'editusertext', data=data)
        try:
            return self._thingify(j['json']['data']['things'][0])
        except Exception:
            raise UnexpectedResponse(j)