def run(self, timeout = 0):
        """
        Run a proactor loop and return new socket events. Timeout is a timedelta
        object, 0 if active coros or None.

        kqueue timeout param is a integer number of nanoseconds (seconds/10**9).
        """
        ptimeout = int(
            timeout.days*86400000000000 +
            timeout.microseconds*1000 +
            timeout.seconds*1000000000
            if timeout else (self.n_resolution if timeout is None else 0)
        )
        if ptimeout>sys.maxint:
            ptimeout = sys.maxint
        if self.tokens:
            events = self.kq.kevent(None, self.default_size, ptimeout)
            # should check here if timeout isn't negative or larger than maxint
            len_events = len(events)-1
            for nr, ev in enumerate(events):
                fd = ev.ident
                act = ev.udata

                if ev.flags & EV_ERROR:
                    ev = EV_SET(fd, act.flags, EV_DELETE)
                    self.kq.kevent(ev)
                    self.handle_error_event(act, 'System error %s.'%ev.data)
                else:
                    if nr == len_events:
                        ret = self.yield_event(act)
                        if not ret:
                            ev.flags = EV_ADD | EV_ENABLE | EV_ONESHOT
                            self.kq.kevent(ev)
                        return ret
                    else:
                        if not self.handle_event(act):
                            ev.flags = EV_ADD | EV_ENABLE | EV_ONESHOT
                            self.kq.kevent(ev)
        else:
            sleep(timeout)