def matches(self, event):
        '''
        Return all matches for this event. The first matcher is also returned for each matched object.
        
        :param event: an input event
        '''
        ret = []
        self._matches(event, set(), ret)
        return tuple(r[0] for r in ret)