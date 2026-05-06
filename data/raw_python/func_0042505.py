def pop_all(self):
        """
        NON-BLOCKING POP ALL IN QUEUE, IF ANY
        """
        with self.lock:
            if self.please_stop:
                return [THREAD_STOP]
            if self.db.status.end == self.start:
                return []

            output = []
            for i in range(self.start, self.db.status.end):
                output.append(self.db[str(i)])

            self.start = self.db.status.end
            return output