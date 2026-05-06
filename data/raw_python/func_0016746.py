def load_children(self):
        """If the Shard doesn't have any children, tries to find some from DescribeStream.

        If the Shard is open this won't find any children, so an empty response doesn't
        mean the Shard will **never** have children.
        """
        # Child count is fixed the first time any of the following happen:
        # 0 :: stream closed or throughput decreased
        # 1 :: shard was open for ~4 hours
        # 2 :: throughput increased

        if self.children:
            return self.children

        # ParentShardId -> [Shard, ...]
        by_parent = collections.defaultdict(list)
        # ShardId -> Shard
        by_id = {}

        for shard in self.session.describe_stream(
                stream_arn=self.stream_arn,
                first_shard=self.shard_id)["Shards"]:
            parent_list = by_parent[shard.get("ParentShardId")]
            shard = Shard(
                stream_arn=self.stream_arn,
                shard_id=shard["ShardId"],
                parent=shard.get("ParentShardId"),
                session=self.session)
            parent_list.append(shard)
            by_id[shard.shard_id] = shard

        # Find this shard when looking up shards by ParentShardId
        by_id[self.shard_id] = self

        # Insert this shard's children, then handle its child's descendants etc.
        to_insert = collections.deque(by_parent[self.shard_id])
        while to_insert:
            shard = to_insert.popleft()
            # ParentShardId -> Shard
            shard.parent = by_id[shard.parent]
            shard.parent.children.append(shard)
            # Continue for any shards that have this shard as their parent
            to_insert.extend(by_parent[shard.shard_id])

        return self.children