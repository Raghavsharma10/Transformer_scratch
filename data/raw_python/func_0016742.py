def token(self):
        """JSON-serializable representation of the current Shard state.

        The token is enough to rebuild the Shard as part of rebuilding a Stream.

        :returns: Shard state as a json-friendly dict
        :rtype: dict
        """
        if self.iterator_type in RELATIVE_ITERATORS:
            logger.warning("creating shard token at non-exact location \"{}\"".format(self.iterator_type))
        token = {
            "stream_arn": self.stream_arn,
            "shard_id": self.shard_id,
            "iterator_type": self.iterator_type,
            "sequence_number": self.sequence_number,
        }
        if self.parent:
            token["parent"] = self.parent.shard_id
        if not self.iterator_type:
            del token["iterator_type"]
        if not self.sequence_number:
            del token["sequence_number"]
        return token