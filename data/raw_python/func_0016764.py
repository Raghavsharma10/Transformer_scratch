def heartbeat(self):
        """Keep active shards with "trim_horizon", "latest" iterators alive by advancing their iterators."""
        for shard in self.active:
            if shard.sequence_number is None:
                records = next(shard)
                # Success!  This shard now has an ``at_sequence`` iterator
                if records:
                    self.buffer.push_all((record, shard) for record in records)
        self.migrate_closed_shards()