def enable_ttl(self, table_name, model):
        """Calls UpdateTimeToLive on the table according to model.Meta["ttl"]

        :param table_name: The name of the table to enable the TTL setting on
        :param model: The model to get TTL settings from
        """
        self._tables.pop(table_name, None)
        ttl_name = model.Meta.ttl["column"].dynamo_name
        request = {
            "TableName": table_name,
            "TimeToLiveSpecification": {"AttributeName": ttl_name, "Enabled": True}
        }
        try:
            self.dynamodb_client.update_time_to_live(**request)
        except botocore.exceptions.ClientError as error:
            raise BloopException("Unexpected error while setting TTL.") from error