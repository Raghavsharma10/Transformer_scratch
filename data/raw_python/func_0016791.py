def get_shard_iterator(self, *, stream_arn, shard_id, iterator_type, sequence_number=None):
        """Wraps :func:`boto3.DynamoDBStreams.Client.get_shard_iterator`.

        :param str stream_arn: Stream arn.  Usually :data:`Shard.stream_arn <bloop.stream.shard.Shard.stream_arn>`.
        :param str shard_id: Shard identifier.  Usually :data:`Shard.shard_id <bloop.stream.shard.Shard.shard_id>`.
        :param str iterator_type: "sequence_at", "sequence_after", "trim_horizon", or "latest"
        :param sequence_number:
        :return: Iterator id, valid for 15 minutes.
        :rtype: str
        :raises bloop.exceptions.RecordsExpired: Tried to get an iterator beyond the Trim Horizon.
        """
        real_iterator_type = validate_stream_iterator_type(iterator_type)
        request = {
            "StreamArn": stream_arn,
            "ShardId": shard_id,
            "ShardIteratorType": real_iterator_type,
            "SequenceNumber": sequence_number
        }
        # boto3 isn't down with literal Nones.
        if sequence_number is None:
            request.pop("SequenceNumber")
        try:
            return self.stream_client.get_shard_iterator(**request)["ShardIterator"]
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "TrimmedDataAccessException":
                raise RecordsExpired from error
            raise BloopException("Unexpected error while creating shard iterator") from error