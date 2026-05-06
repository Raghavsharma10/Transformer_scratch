def pop_chunk(self, chunk_max_size):
        """Pops a chunk of the given max size.

        Optimized to avoid too much string copies.

        Args:
            chunk_max_size (int): max size of the returned chunk.

        Returns:
            string (bytes) with a size <= chunk_max_size.
        """
        if self._total_length < chunk_max_size:
            # fastpath (the whole queue fit in a single chunk)
            res = self._tobytes()
            self.clear()
            return res
        first_iteration = True
        while True:
            try:
                data = self._deque.popleft()
                data_length = len(data)
                self._total_length -= data_length
                if first_iteration:
                    # first iteration
                    if data_length == chunk_max_size:
                        # we are lucky !
                        return data
                    elif data_length > chunk_max_size:
                        # we have enough data at first iteration
                        # => fast path optimization
                        view = self._get_pointer_or_memoryview(data,
                                                               data_length)
                        self.appendleft(view[chunk_max_size:])
                        return view[:chunk_max_size]
                    else:
                        # no single iteration fast path optimization :-(
                        # let's use a WriteBuffer to build the result chunk
                        chunk_write_buffer = WriteBuffer()
                else:
                    # not first iteration
                    if chunk_write_buffer._total_length + data_length \
                       > chunk_max_size:
                        view = self._get_pointer_or_memoryview(data,
                                                               data_length)
                        limit = chunk_max_size - \
                            chunk_write_buffer._total_length - data_length
                        self.appendleft(view[limit:])
                        data = view[:limit]
                chunk_write_buffer.append(data)
                if chunk_write_buffer._total_length >= chunk_max_size:
                    break
            except IndexError:
                # the buffer is empty (so no memoryview inside)
                self._has_view = False
                break
            first_iteration = False
        return chunk_write_buffer._tobytes()