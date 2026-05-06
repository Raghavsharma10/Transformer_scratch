def run(self, timeout = 0):
        """
        Run a proactor loop and return new socket events. Timeout is a timedelta
        object, 0 if active coros or None.

        epoll timeout param is a integer number of miliseconds (seconds/1000).
        """
        ptimeout = int(timeout.microseconds/1000+timeout.seconds*1000
                if timeout else (self.m_resolution if timeout is None else 0))
        if self.tokens:
            epoll_fd = self.epoll_fd
            events = epoll_wait(epoll_fd, 1024, ptimeout)
            len_events = len(events)-1
            for nr, (ev, fd) in enumerate(events):
                act = self.shadow.pop(fd)
                if ev & EPOLLHUP:
                    epoll_ctl(self.epoll_fd, EPOLL_CTL_DEL, fd, 0)
                    self.handle_error_event(act, 'Hang up.', ConnectionClosed)
                elif ev & EPOLLERR:
                    epoll_ctl(self.epoll_fd, EPOLL_CTL_DEL, fd, 0)
                    self.handle_error_event(act, 'Unknown error.')
                else:
                    if nr == len_events:
                        ret = self.yield_event(act)
                        if not ret:
                            epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, ev | EPOLLONESHOT)
                            self.shadow[fd] = act
                        return ret
                    else:
                        if not self.handle_event(act):
                            epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, ev | EPOLLONESHOT)
                            self.shadow[fd] = act

        else:
            sleep(timeout)