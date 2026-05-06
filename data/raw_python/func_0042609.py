def pop_all(self, priority=None):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        output = []
        with self.lock:
            if not priority:
                priority = self.highest_entry()
            if priority:
                output = list(self.queue[priority].queue)
                self.queue[priority].queue.clear()
        return output