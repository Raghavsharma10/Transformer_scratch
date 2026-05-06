def compact(self):
        """Compacts the queue: removes all the messages from the queue that
        have been fetched by all the subscribed coroutines.
        Returns the number of messages that have been removed."""
        if self.subscribers:
            level = min(self.subscribers.itervalues())
            if level:
                del self.messages[:level]
            return level
        else:
            level = len(self.messages)
            del self.messages[:]
            return level