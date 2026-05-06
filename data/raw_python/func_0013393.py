def isFull(self):
        """
        Returns True if the response buffer is full, and False otherwise.
        The buffer is full if either (1) the number of items in the value
        list is >= pageSize or (2) the total length of the serialised
        elements in the page is >= maxBufferSize.

        If page_size or max_response_length were not set in the request
        then they're not checked.
        """
        return (
            (self._pageSize > 0 and self._numElements >= self._pageSize) or
            (self._bufferSize >= self._maxBufferSize)
        )