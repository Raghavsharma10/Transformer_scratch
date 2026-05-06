def unpack_shards(shards, stream_arn, session):
    """List[Dict] -> Dict[shard_id, Shard].

    Each Shards' parent/children are hooked up with the other Shards in the list.
    """
    if not shards:
        return {}

    # When unpacking tokens, shard id key is "shard_id"
    # When unpacking DescribeStream responses, shard id key is "ShardId"
    if "ShardId" in shards[0]:
        shards = _translate_shards(shards)

    by_id = {shard_token["shard_id"]:
             Shard(stream_arn=stream_arn, shard_id=shard_token["shard_id"],
                   iterator_type=shard_token.get("iterator_type"), sequence_number=shard_token.get("sequence_number"),
                   parent=shard_token.get("parent"), session=session)
             for shard_token in shards}

    for shard in by_id.values():
        if shard.parent:
            shard.parent = by_id[shard.parent]
            shard.parent.children.append(shard)
    return by_id