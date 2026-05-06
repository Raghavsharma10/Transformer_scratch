def add_new(self, command):
        """Add a new entry to the queue."""
        self.queue[self.next_key] = command
        self.queue[self.next_key]['status'] = 'queued'
        self.queue[self.next_key]['returncode'] = ''
        self.queue[self.next_key]['stdout'] = ''
        self.queue[self.next_key]['stderr'] = ''
        self.queue[self.next_key]['start'] = ''
        self.queue[self.next_key]['end'] = ''

        self.next_key += 1
        self.write()