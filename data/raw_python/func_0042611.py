def pop_one(self, priority=None):
        """
        NON-BLOCKING POP IN QUEUE, IF ANY
        """
        with self.lock:
            if not priority:
                priority = self.highest_entry()
            if self.closed:
                return [THREAD_STOP]
            elif not self.queue:
                return None
            else:
                v =self.pop(priority=priority)
                if v is THREAD_STOP:  # SENDING A STOP INTO THE QUEUE IS ALSO AN OPTION
                    self.closed.go()
                return v