def pop_all_queues(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        output = []
        with self.lock:
            for q in self.queue:
                output.extend(list(q.queue))
                q.queue.clear()

        return output