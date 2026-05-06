def _move_stream_endpoint(coordinator, position):
    """Move to the "trim_horizon" or "latest" of the entire stream."""
    # 0) Everything will be rebuilt from DescribeStream.
    stream_arn = coordinator.stream_arn
    coordinator.roots.clear()
    coordinator.active.clear()
    coordinator.buffer.clear()

    # 1) Build a Dict[str, Shard] of the current Stream from a DescribeStream call
    current_shards = coordinator.session.describe_stream(stream_arn=stream_arn)["Shards"]
    current_shards = unpack_shards(current_shards, stream_arn, coordinator.session)

    # 2) Roots are any shards without parents.
    coordinator.roots.extend(shard for shard in current_shards.values() if not shard.parent)

    # 3.0) Stream trim_horizon is the combined trim_horizon of all roots.
    if position == "trim_horizon":
        for shard in coordinator.roots:
            shard.jump_to(iterator_type="trim_horizon")
        coordinator.active.extend(coordinator.roots)
    # 3.1) Stream latest is the combined latest of all shards without children.
    else:
        for root in coordinator.roots:
            for shard in root.walk_tree():
                if not shard.children:
                    shard.jump_to(iterator_type="latest")
                    coordinator.active.append(shard)