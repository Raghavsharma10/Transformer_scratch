def advance_shards(self):
        """Poll active shards for records and insert them into the buffer.  Rotate exhausted shards.

        Returns immediately if the buffer isn't empty.
        """
        # Don't poll shards when there are pending records.
        if self.buffer:
            return

        # 0) Collect new records from all active shards.
        record_shard_pairs = []
        for shard in self.active:
            records = next(shard)
            if records:
                record_shard_pairs.extend((record, shard) for record in records)
        self.buffer.push_all(record_shard_pairs)

        self.migrate_closed_shards()