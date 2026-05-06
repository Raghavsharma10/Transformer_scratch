def process_op(self, op, coro):
        "Process a (op, coro) pair and return another pair. Handles exceptions."
        if op is None:
            if self.active:
                self.active.append((op, coro))
            else:
                return op, coro
        else:
            try:
                result = op.process(self, coro) or (None, None)
            except:
                op.state = events.ERRORED
                result = CoroutineException(*sys.exc_info()), coro
            return result
        return None, None