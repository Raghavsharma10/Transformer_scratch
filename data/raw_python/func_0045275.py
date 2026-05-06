def run(self, timeout = 0):
        """
        Run a proactor loop and return new socket events. Timeout is a timedelta
        object, 0 if active coros or None.

        kqueue timeout param is a integer number of nanoseconds (seconds/10**9).
        """
        ptimeout = float(
            timeout.microseconds/1000000+timeout.seconds if timeout
            else (self.resolution if timeout is None else 0)
        )
        if self.tokens:
            events = self.kcontrol(None, self.default_size, ptimeout)
            len_events = len(events)-1
            for nr, ev in enumerate(events):
                fd = ev.ident
                act = self.shadow.pop(fd)

                if ev.flags & KQ_EV_ERROR:
                    self.kcontrol((kevent(fd, act.flags, KQ_EV_DELETE),), 0)
                    self.handle_error_event(act, 'System error %s.'%ev.data)
                else:
                    if nr == len_events:
                        ret = self.yield_event(act)
                        if not ret:
                            ev.flags = KQ_EV_ADD | KQ_EV_ONESHOT
                            self.kcontrol((ev,), 0)
                            self.shadow[fd] = act
                        return ret
                    else:
                        if not self.handle_event(act):
                            ev.flags = KQ_EV_ADD | KQ_EV_ONESHOT
                            self.kcontrol((ev,), 0)
                            self.shadow[fd] = act
        else:
            sleep(timeout)