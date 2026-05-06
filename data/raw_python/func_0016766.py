def remove_shard(self, shard, drop_buffered_records=False):
        """Remove a Shard from the Coordinator.  Drops all buffered records from the Shard.

        If the Shard is active or a root, it is removed and any children promoted to those roles.

        :param shard: The shard to remove
         :type shard: :class:`~bloop.stream.shard.Shard`
        :param bool drop_buffered_records:
            Whether records from this shard should be removed.
            Default is False.
        """
        try:
            self.roots.remove(shard)
        except ValueError:
            # Wasn't a root Shard
            pass
        else:
            self.roots.extend(shard.children)

        try:
            self.active.remove(shard)
        except ValueError:
            # Wasn't an active Shard
            pass
        else:
            self.active.extend(shard.children)

        if drop_buffered_records:
            # TODO can this be improved?  Gets expensive for high-volume streams with large buffers
            heap = self.buffer.heap
            # Clear buffered records from the shard.  Each record is (ordering, record, shard)
            to_remove = [x for x in heap if x[2] is shard]
            for x in to_remove:
                heap.remove(x)