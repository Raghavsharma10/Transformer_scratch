def run(self, timeout = 0):
        """
        Calls GetQueuedCompletionStatus and handles completion via
        IOCPProactor.process_op.
        """
        # same resolution as epoll
        ptimeout = int(
            timeout.days * 86400000 +
            timeout.microseconds / 1000 +
            timeout.seconds * 1000
            if timeout else (self.m_resolution if timeout is None else 0)
        )
        if self.tokens:
            scheduler = self.scheduler
            urgent = None
            # we use urgent as a optimisation: the last operation is returned
            #directly to the scheduler (the sched might just run it till it
            #goes to sleep) and not added in the sched.active queue
            while 1:
                try:
                    rc, nbytes, key, overlap = win32file.GetQueuedCompletionStatus(
                        self.iocp,
                        0 if urgent else ptimeout
                    )
                except RuntimeError:
                    # we will get "This overlapped object has lost all its
                    # references so was destroyed" when we remove a operation,
                    # it is garbage collected and the overlapped completes
                    # afterwards
                    break

                # well, this is a bit weird, if we get a aborted rc (via CancelIo
                #i suppose) evaluating the overlap crashes the interpeter
                #with a memory read error
                if rc != win32file.WSA_OPERATION_ABORTED and overlap:

                    if urgent:
                        op, coro = urgent
                        urgent = None
                        if op.prio & priority.OP:
                            # imediately run the asociated coroutine step
                            op, coro = scheduler.process_op(
                                coro.run_op(op, scheduler),
                                coro
                            )
                        if coro:
                            #TODO, what "op and "
                            if op and (op.prio & priority.CORO):
                                scheduler.active.appendleft( (op, coro) )
                            else:
                                scheduler.active.append( (op, coro) )
                    if overlap.object:
                        urgent = self.process_op(rc, nbytes, overlap)
                else:
                    break
            return urgent
        else:
            sleep(timeout)