def heap_item(clock, record, shard):
    """Create a tuple of (ordering, (record, shard)) for use in a RecordBuffer."""
    # Primary ordering is by event creation time.
    # However, creation time is *approximate* and has whole-second resolution.
    # This means two events in the same shard within one second can't be ordered.
    ordering = record["meta"]["created_at"]
    # From testing, SequenceNumber isn't a guaranteed ordering either.  However,
    # it is guaranteed to be unique within a shard.  This will be tie-breaker
    # for multiple records within the same shard, within the same second.
    second_ordering = int(record["meta"]["sequence_number"])
    # It's possible though unlikely, that sequence numbers will collide across
    # multiple shards, within the same second.  The final tie-breaker is
    # a monotonically increasing integer from the buffer.
    total_ordering = (ordering, second_ordering, clock())
    return total_ordering, record, shard