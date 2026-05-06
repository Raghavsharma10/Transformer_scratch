def run(self, timeout = 0):
        """
        Run a proactor loop and return new socket events. Timeout is a timedelta
        object, 0 if active coros or None.
        """
        # poll timeout param is a integer number of miliseconds (seconds/1000).
        ptimeout = int(
            timeout.days * 86400000 +
            timeout.microseconds / 1000 +
            timeout.seconds * 1000
            if timeout else (self.m_resolution if timeout is None else 0)
        )
        if self.tokens:
            events = self.poller.poll(ptimeout)
            len_events = len(events)-1
            for nr, (fd, ev) in enumerate(events):
                act = self.shadow.pop(fd)
                if ev & POLLHUP:
                    self.poller.unregister(fd)
                    self.handle_error_event(act, 'Hang up.', ConnectionClosed)
                elif ev & POLLNVAL:
                    self.poller.unregister(fd)
                    self.handle_error_event(act, 'Invalid descriptor.')
                elif ev & POLLERR:
                    self.poller.unregister(fd)
                    self.handle_error_event(act, 'Unknown error.')
                else:
                    if nr == len_events:
                        ret = self.yield_event(act)
                        if ret:
                            self.poller.unregister(fd)
                        else:
                            self.shadow[fd] = act
                        return ret
                    else:
                        if self.handle_event(act):
                            self.poller.unregister(fd)
                        else:
                            self.shadow[fd] = act

        else:
            sleep(timeout)