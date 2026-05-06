def get_stream_records(self, iterator_id):
        """Wraps :func:`boto3.DynamoDBStreams.Client.get_records`.

        :param iterator_id: Iterator id.  Usually :data:`Shard.iterator_id <bloop.stream.shard.Shard.iterator_id>`.
        :return: Dict with "Records" list (may be empty) and "NextShardIterator" str (may not exist).
        :rtype: dict
        :raises bloop.exceptions.RecordsExpired: The iterator moved beyond the Trim Horizon since it was created.
        :raises bloop.exceptions.ShardIteratorExpired: The iterator was created more than 15 minutes ago.
        """
        try:
            return self.stream_client.get_records(ShardIterator=iterator_id)
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "TrimmedDataAccessException":
                raise RecordsExpired from error
            elif error.response["Error"]["Code"] == "ExpiredIteratorException":
                raise ShardIteratorExpired from error
            raise BloopException("Unexpected error while getting records.") from error