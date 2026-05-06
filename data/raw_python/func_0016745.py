def seek_to(self, position):
        """Move the Shard's iterator to the earliest record after the :class:`~datetime.datetime` time.

        Returns the first records at or past ``position``.  If the list is empty,
        the seek failed to find records, either because the Shard is exhausted or it
        reached the HEAD of an open Shard.

        :param position: The position in time to move to.
        :type position: :class:`~datetime.datetime`
        :returns: A list of the first records found after ``position``.  May be empty.
        """
        # 0) We have no way to associate the date with a position,
        #    so we have to scan the shard from the beginning.
        self.jump_to(iterator_type="trim_horizon")

        position = int(position.timestamp())

        while (not self.exhausted) and (self.empty_responses < CALLS_TO_REACH_HEAD):
            records = self.get_records()
            # We can skip the whole record set if the newest (last) record isn't new enough.
            if records and records[-1]["meta"]["created_at"].timestamp() >= position:
                # Looking for the first number *below* the position.
                for offset, record in enumerate(reversed(records)):
                    if record["meta"]["created_at"].timestamp() < position:
                        index = len(records) - offset
                        return records[index:]
                return records

        # Either exhausted the Shard or caught up to HEAD.
        return []