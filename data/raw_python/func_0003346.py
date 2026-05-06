async def wait_with_timeout(self, timeout, *matchers):
        """
        Wait for multiple event matchers, or until timeout.
        
        :param timeout: a timeout value
        
        :param \*matchers: event matchers
        
        :return: (is_timeout, event, matcher). When is_timeout = True, event = matcher = None.
        """
        if timeout is None:
            ev, m = await M_(*matchers)
            return False, ev, m
        else:
            th = self.scheduler.setTimer(timeout)
            try:
                tm = TimerEvent.createMatcher(th)
                ev, m = await M_(*(tuple(matchers) + (tm,)))
                if m is tm:
                    return True, None, None
                else:
                    return False, ev, m
            finally:
                self.scheduler.cancelTimer(th)