def jump_to(self, *, iterator_type, sequence_number=None):
        """Move to a new position in the shard using the standard parameters to GetShardIterator.

        :param str iterator_type: "trim_horizon", "at_sequence", "after_sequence", "latest"
        :param str sequence_number: *(Optional)* Sequence number to use with at/after sequence.  Default is None.
        """
        # Just a simple wrapper; let the caller handle RecordsExpired
        self.iterator_id = self.session.get_shard_iterator(
            stream_arn=self.stream_arn,
            shard_id=self.shard_id,
            iterator_type=iterator_type,
            sequence_number=sequence_number)
        self.iterator_type = iterator_type
        self.sequence_number = sequence_number
        self.empty_responses = 0