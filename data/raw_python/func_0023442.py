def remove(self, key):
        """Remove a key from the queue, return `False` if no such key exists."""
        if key in self.queue:
            del self.queue[key]
            self.write()
            return True
        return False