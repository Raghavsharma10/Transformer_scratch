def push(self, el):
        """ Put a new element in the queue. """
        count = next(self.counter)
        heapq.heappush(self._queue, (el, count))