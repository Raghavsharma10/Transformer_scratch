def run(self, timeout = 0):
        """
        Run a proactor loop and return new socket events. Timeout is a timedelta
        object, 0 if active coros or None.

        epoll timeout param is a integer number of seconds.
        """
        ptimeout = float(
            timeout.microseconds/1000000+timeout.seconds if timeout
            else (self.resolution if timeout is None else 0)
        )
        if self.tokens:
            events = self.epoll_obj.poll(ptimeout, 1024)
            len_events = len(events)-1
            for nr, (fd, ev) in enumerate(events):
                act = self.shadow.pop(fd)
                if ev & EPOLLHUP:
                    self.epoll_obj.unregister(fd)
                    self.handle_error_event(act, 'Hang up.', ConnectionClosed)
                elif ev & EPOLLERR:
                    self.epoll_obj.unregister(fd)
                    self.handle_error_event(act, 'Unknown error.')
                else:
                    if nr == len_events:
                        ret = self.yield_event(act)
                        if not ret:
                            self.epoll_obj.modify(fd, ev | EPOLLONESHOT)
                            self.shadow[fd] = act
                        return ret
                    else:
                        if not self.handle_event(act):
                            self.epoll_obj.modify(fd, ev | EPOLLONESHOT)
                            self.shadow[fd] = act


        else:
            sleep(timeout)