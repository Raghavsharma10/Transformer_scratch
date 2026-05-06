def push_all(self, record_shard_pairs):
        """Push multiple (record, shard) pairs at once, with only one :meth:`heapq.heapify` call to maintain order.

        :param record_shard_pairs: list of ``(record, shard)`` tuples
            (see :func:`~bloop.stream.buffer.RecordBuffer.push`).
        """
        # Faster than inserting one at a time; the heap is sorted once after all inserts.
        for record, shard in record_shard_pairs:
            item = heap_item(self.clock, record, shard)
            self.heap.append(item)
        heapq.heapify(self.heap)