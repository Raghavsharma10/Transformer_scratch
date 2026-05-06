def vote(self, id_, dir_):
        """Login required.  POSTs a vote.  Returns True or raises :class:`exceptions.UnexpectedResponse` if non-"truthy" value in response.
        
        See https://github.com/reddit/reddit/wiki/API%3A-vote.
        
        URL: ``http://www.reddit.com/api/vote/``
        
        :param id\_: full id of object voting on
        :param dir\_: direction of vote (1, 0, or -1)
        """
        data = dict(id=id_, dir=dir_)
        j = self.post('api', 'vote', data=data)
        return assert_truthy(j)