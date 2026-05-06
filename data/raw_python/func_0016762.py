def _move_stream_token(coordinator, token):
    """Move to the Stream position described by the token.

    The following rules are applied when interpolation is required:
    - If a shard does not exist (past the trim_horizon) it is ignored.  If that
      shard had children, its children are also checked against the existing shards.
    - If none of the shards in the token exist, then InvalidStream is raised.
    - If a Shard expects its iterator to point to a SequenceNumber that is now past
      that Shard's trim_horizon, the Shard instead points to trim_horizon.
    """
    stream_arn = coordinator.stream_arn = token["stream_arn"]
    # 0) Everything will be rebuilt from the DescribeStream masked by the token.
    coordinator.roots.clear()
    coordinator.active.clear()
    coordinator.closed.clear()
    coordinator.buffer.clear()

    # Injecting the token gives us access to the standard shard management functions
    token_shards = unpack_shards(token["shards"], stream_arn, coordinator.session)
    coordinator.roots = [shard for shard in token_shards.values() if not shard.parent]
    coordinator.active.extend(token_shards[shard_id] for shard_id in token["active"])

    # 1) Build a Dict[str, Shard] of the current Stream from a DescribeStream call
    current_shards = coordinator.session.describe_stream(stream_arn=stream_arn)["Shards"]
    current_shards = unpack_shards(current_shards, stream_arn, coordinator.session)

    # 2) Trying to find an intersection with the actual Stream by walking each root shard's tree.
    #    Prune any Shard with no children that's not part of the actual Stream.
    #    Raise InvalidStream if the entire token is pruned.
    unverified = collections.deque(coordinator.roots)
    while unverified:
        shard = unverified.popleft()
        if shard.shard_id not in current_shards:
            logger.info("Unknown or expired shard \"{}\" - pruning from stream token".format(shard.shard_id))
            coordinator.remove_shard(shard, drop_buffered_records=True)
            unverified.extend(shard.children)

    # 3) Everything was pruned, so the token describes an unknown stream.
    if not coordinator.roots:
        raise InvalidStream("This token has no relation to the actual Stream.")

    # 4) Now that everything's verified, grab new iterators for the coordinator's active Shards.
    for shard in coordinator.active:
        try:
            if shard.iterator_type is None:
                # Descendant of an unknown shard
                shard.iterator_type = "trim_horizon"
            # Move back to the token's specified position
            shard.jump_to(iterator_type=shard.iterator_type, sequence_number=shard.sequence_number)
        except RecordsExpired:
            # This token shard's sequence_number is beyond the trim_horizon.
            # The next closest record is at trim_horizon.
            msg = "SequenceNumber \"{}\" in shard \"{}\" beyond trim horizon: jumping to trim_horizon"
            logger.info(msg.format(shard.sequence_number, shard.shard_id))
            shard.jump_to(iterator_type="trim_horizon")