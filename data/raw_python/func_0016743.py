def walk_tree(self):
        """Generator that yields each :class:`~bloop.stream.shard.Shard` by walking the shard's children in order."""
        shards = collections.deque([self])
        while shards:
            shard = shards.popleft()
            yield shard
            shards.extend(shard.children)