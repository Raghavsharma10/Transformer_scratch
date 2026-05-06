def queue_delete(self, queue, if_unused=False, if_empty=False):
        """Delete queue by name."""
        return self.channel.queue_delete(queue, if_unused, if_empty)