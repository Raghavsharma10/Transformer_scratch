def handle_event(self, act):
        """
        Handle completion for a request.

        Calls the scheduler to run or schedule the associated coroutine.
        """
        scheduler = self.scheduler
        if act in self.tokens:
            coro = act.coro
            op = self.try_run_act(act, self.tokens[act])
            if op:
                del self.tokens[act]
                if scheduler.ops_greedy:
                    while True:
                        op, coro = scheduler.process_op(coro.run_op(op, scheduler), coro)
                        if not op and not coro:
                            break
                else:
                    if op.prio & priority.OP:
                        op, coro = scheduler.process_op(coro.run_op(op, scheduler), coro)
                    if coro and op:
                        if op.prio & priority.CORO:
                            scheduler.active.appendleft( (op, coro) )
                        else:
                            scheduler.active.append( (op, coro) )
            else:
                return
        else:
            import warnings
            warnings.warn("Got event for unkown act: %s" % act)
        return True