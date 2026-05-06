def pop(self, till=None, priority=None):
        """
        WAIT FOR NEXT ITEM ON THE QUEUE
        RETURN THREAD_STOP IF QUEUE IS CLOSED
        RETURN None IF till IS REACHED AND QUEUE IS STILL EMPTY

        :param till:  A `Signal` to stop waiting and return None
        :return:  A value, or a THREAD_STOP or None
        """
        if till is not None and not isinstance(till, Signal):
            Log.error("expecting a signal")

        with self.lock:
            while True:
                if not priority:
                    priority = self.highest_entry()
                if priority:
                    value = self.queue[priority].queue.popleft()
                    return value
                if self.closed:
                    break
                if not self.lock.wait(till=till | self.closed):
                    if self.closed:
                        break
                    return None
        (DEBUG or not self.silent) and Log.note(self.name + " queue stopped")
        return THREAD_STOP