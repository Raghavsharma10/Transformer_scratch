def pop_one(self):
        """
        NON-BLOCKING POP IN QUEUE, IF ANY
        """
        with self.lock:
            if self.closed:
                return [THREAD_STOP]
            elif not self.queue:
                return None
            else:
                v =self.queue.pop()
                if v is THREAD_STOP:  # SENDING A STOP INTO THE QUEUE IS ALSO AN OPTION
                    self.closed.go()
                return v