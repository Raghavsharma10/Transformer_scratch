def expire(self, current_time=None):
        """ Expire any old entries

        @param current_time: Optional time to be used to clean up queue (can be
                             used in unit tests)
        """
        if self._queue.is_empty():
            return

        if current_time is None:
            current_time = time.time()

        while not self._queue.is_empty():
            # Get top most item
            top = self._queue.peek()

            # Early exit if item was not promoted and its expiration time
            # is greater than now.
            if top.promoted is None and top.expiry_date > current_time:
                break

            # Pop item from the stack
            top = self._queue.pop()

            need_reschedule = (top.promoted is not None
                               and top.promoted > current_time)

            # Give chance to reschedule
            if not need_reschedule:
                top.promoted = None
                top.on_delete(False)

                need_reschedule = (top.promoted is not None
                                   and top.promoted > current_time)

            # If item is promoted and expiration time somewhere in future
            # just reschedule it
            if need_reschedule:
                top.expiry_date = top.promoted
                top.promoted = None
                self._queue.push(top)
            else:
                del self._items[top.session_id]