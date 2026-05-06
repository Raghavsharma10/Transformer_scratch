async def wait_for_all(self, *matchers, eventlist = None, eventdict = None, callback = None):
        """
        Wait until each matcher matches an event. When this coroutine method returns,
        `eventlist` is set to the list of events in the arriving order (may not
        be the same as the matchers); `eventdict` is set to a dictionary
        `{matcher1: event1, matcher2: event2, ...}`
        
        :param eventlist: use external event list, so when an exception occurs
                          (e.g. routine close), you can retrieve the result
                          from the passed-in list
        
        :param eventdict: use external event dict
        
        :param callback: if not None, the callback should be a callable callback(event, matcher)
                         which is called each time an event is received
        
        :return: (eventlist, eventdict)
        """
        if eventdict is None:
            eventdict = {}
        if eventlist is None:
            eventlist = []
        ms = len(matchers)
        last_matchers = Diff_(matchers)
        while ms:
            ev, m = await last_matchers
            ms -= 1
            if callback:
                callback(ev, m)
            eventlist.append(ev)
            eventdict[m] = ev
            last_matchers = Diff_(last_matchers, remove=(m,))
        return eventlist, eventdict