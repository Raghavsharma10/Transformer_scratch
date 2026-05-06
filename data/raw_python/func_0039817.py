def distinguish(self, id_, how=True):
        """Login required.  Sends POST to distinguish a submission or comment.  Returns :class:`things.Link` or :class:`things.Comment`, or raises :class:`exceptions.UnexpectedResponse` otherwise.
        
        URL: ``http://www.reddit.com/api/distinguish/``
        
        :param id\_: full id of object to distinguish
        :param how: either True, False, or 'admin'
        """
        if how == True:
            h = 'yes'
        elif how == False:
            h = 'no'
        elif how == 'admin':
            h = 'admin'
        else:
            raise ValueError("how must be either True, False, or 'admin'") 
        data = dict(id=id_)
        j = self.post('api', 'distinguish', h, data=data)
        try:
            return self._thingify(j['json']['data']['things'][0])
        except Exception:
            raise UnexpectedResponse(j)