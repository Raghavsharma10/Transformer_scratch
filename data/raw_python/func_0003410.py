def matchesWithMatchers(self, event):
        '''
        Return all matches for this event. The first matcher is also returned for each matched object.
        
        :param event: an input event
        '''
        ret = []
        self._matches(event, set(), ret)
        return tuple(ret)