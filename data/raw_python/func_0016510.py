def queue_purge(self, queue, **kwargs):
        """Discard all messages in the queue. This will delete the messages
        and results in an empty queue."""
        return self.channel.queue_purge(queue=queue).message_count