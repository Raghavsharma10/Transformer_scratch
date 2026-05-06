def push(self, record, shard):
        """Push a new record into the buffer

        :param dict record: new record
        :param shard: Shard the record came from
        :type shard: :class:`~bloop.stream.shard.Shard`
        """
        heapq.heappush(self.heap, heap_item(self.clock, record, shard))