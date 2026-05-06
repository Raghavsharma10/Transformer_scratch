def clear(self):
        """Remove all completed tasks from the queue."""
        for key in list(self.queue.keys()):
            if self.queue[key]['status'] in ['done', 'failed']:
                del self.queue[key]
        self.write()