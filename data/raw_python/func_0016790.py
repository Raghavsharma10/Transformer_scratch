def describe_stream(self, stream_arn, first_shard=None):
        """Wraps :func:`boto3.DynamoDBStreams.Client.describe_stream`, handling continuation tokens.

        :param str stream_arn: Stream arn, usually from the model's ``Meta.stream["arn"]``.
        :param str first_shard: *(Optional)* If provided, only shards after this shard id will be returned.
        :return: All shards in the stream, or a subset if ``first_shard`` is provided.
        :rtype: dict
        """
        description = {"Shards": []}

        request = {"StreamArn": stream_arn, "ExclusiveStartShardId": first_shard}
        # boto3 isn't down with literal Nones.
        if first_shard is None:
            request.pop("ExclusiveStartShardId")

        while request.get("ExclusiveStartShardId") is not missing:
            try:
                response = self.stream_client.describe_stream(**request)["StreamDescription"]
            except botocore.exceptions.ClientError as error:
                if error.response["Error"]["Code"] == "ResourceNotFoundException":
                    raise InvalidStream(f"The stream arn {stream_arn!r} does not exist.") from error
                raise BloopException("Unexpected error while describing stream.") from error
            # Docs aren't clear if the terminal value is null, or won't exist.
            # Since we don't terminate the loop on None, the "or missing" here
            # will ensure we stop on a falsey value.
            request["ExclusiveStartShardId"] = response.pop("LastEvaluatedShardId", None) or missing
            description["Shards"].extend(response.pop("Shards", []))
            description.update(response)
        return description