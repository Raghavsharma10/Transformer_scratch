def handle_timeouts(self):
        """Handle timeouts. Raise timeouted operations with a OperationTimeout
        in the associated coroutine (if they are still alive and the operation
        hasn't actualy sucessfuly completed) or, if the operation has a
        weak_timeout flag, update the timeout point and add it back in the
        heapq.

        weak_timeout notes:

        * weak_timeout means a last_update attribute is updated with
          a timestamp of the last activity in the operation - for example, a
          may recieve new data and not complete (not enough data, etc)
        * if there was activity since the last time we've cheched this
          timeout we push it back in the heapq with a timeout value we'll check
          it again

        Also, we call a cleanup on the op, only if cleanup return true we raise
        the timeout (finalized isn't enough to check if the op has completed
        since finalized is set when the operation gets back in the coro - and
        it might still be in the Scheduler.active queue when we get to this
        timeout - well, this is certainly a problem magnet: TODO: fix_finalized)
        """
        now = getnow()
        #~ print '>to:', self.timeouts, self.timeouts and self.timeouts[0].timeout <= now
        while self.timeouts and self.timeouts[0].timeout <= now:
            op = heapq.heappop(self.timeouts)

            coro = op.coro
            if op.weak_timeout and hasattr(op, 'last_update'):
                if op.last_update > op.last_checkpoint:
                    op.last_checkpoint = op.last_update
                    op.timeout = op.last_checkpoint + op.delta
                    heapq.heappush(self.timeouts, op)
                    continue

            if op.state is events.RUNNING and coro and coro.running and \
                                                    op.cleanup(self, coro):

                self.active.append((
                    CoroutineException(
                        events.OperationTimeout,
                        events.OperationTimeout(op)
                    ),
                    coro
                ))