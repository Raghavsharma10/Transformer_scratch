def _move_stream_time(coordinator, time):
    """Scan through the *entire* Stream for the first record after ``time``.

    This is an extremely expensive, naive algorithm that starts at trim_horizon and simply
    dumps records into the void until the first hit.  General improvements in performance are
    tough; we can use the fact that Shards have a max life of 24hr to pick a pretty-good starting
    point for any Shard trees with 6 generations.  Even then we can't know how close the oldest one
    is to rolling off so we either hit trim_horizon, or iterate an extra Shard more than we need to.

    The corner cases are worse; short trees, recent splits, trees with different branch heights.
    """
    if time > datetime.datetime.now(datetime.timezone.utc):
        _move_stream_endpoint(coordinator, "latest")
        return

    _move_stream_endpoint(coordinator, "trim_horizon")
    shard_trees = collections.deque(coordinator.roots)
    while shard_trees:
        shard = shard_trees.popleft()
        records = shard.seek_to(time)

        # Success!  This section of some Shard tree is at the desired time.
        if records:
            coordinator.buffer.push_all((record, shard) for record in records)

        # Closed shard, keep searching its children.
        elif shard.exhausted:
            coordinator.remove_shard(shard, drop_buffered_records=True)
            shard_trees.extend(shard.children)