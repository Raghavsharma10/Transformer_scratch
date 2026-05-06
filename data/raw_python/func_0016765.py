def token(self):
        """JSON-serializable representation of the current Stream state.

        Use :func:`Engine.stream(YourModel, token) <bloop.engine.Engine.stream>` to create an identical stream,
        or :func:`stream.move_to(token) <bloop.stream.Stream.move_to>` to move an existing stream to this position.

        :returns: Stream state as a json-friendly dict
        :rtype: dict
        """
        # 0) Trace roots and active shards
        active_ids = []
        shard_tokens = []
        for root in self.roots:
            for shard in root.walk_tree():
                shard_tokens.append(shard.token)
                # dedupe, stream_arn will be in the root token
                shard_tokens[-1].pop("stream_arn")
        active_ids.extend((shard.shard_id for shard in self.active))

        # 1) Inject closed shards
        for shard in self.closed.keys():
            active_ids.append(shard.shard_id)
            shard_tokens.append(shard.token)
            shard_tokens[-1].pop("stream_arn")

        return {
            "stream_arn": self.stream_arn,
            "active": active_ids,
            "shards": shard_tokens
        }