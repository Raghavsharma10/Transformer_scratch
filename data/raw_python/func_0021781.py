def list_kinesis_applications(region, filter_by_kwargs):
    """List all the kinesis applications along with the shards for each stream"""
    conn = boto.kinesis.connect_to_region(region)
    streams = conn.list_streams()['StreamNames']
    kinesis_streams = {}
    for stream_name in streams:
        shard_ids = []
        shards = conn.describe_stream(stream_name)['StreamDescription']['Shards']
        for shard in shards:
            shard_ids.append(shard['ShardId'])
        kinesis_streams[stream_name] = shard_ids
    return kinesis_streams