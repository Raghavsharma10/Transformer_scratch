def queue_purge(self, queue, **kwargs):
        """Discard all messages in the queue."""
        qsize = mqueue.qsize()
        mqueue.queue.clear()
        return qsize