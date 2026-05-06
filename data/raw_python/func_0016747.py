def get_records(self):
        """Get the next set of records in this shard.  An empty list doesn't guarantee the shard is exhausted.

        :returns: A list of reformatted records.  May be empty.
        """
        # Won't be able to find new records.
        if self.exhausted:
            return []

        # Already caught up, just the one call please.
        if self.empty_responses >= CALLS_TO_REACH_HEAD:
            return self._apply_get_records_response(self.session.get_stream_records(self.iterator_id))

        # Up to 5 calls to try and find a result
        while self.empty_responses < CALLS_TO_REACH_HEAD and not self.exhausted:
            records = self._apply_get_records_response(self.session.get_stream_records(self.iterator_id))
            if records:
                return records

        return []