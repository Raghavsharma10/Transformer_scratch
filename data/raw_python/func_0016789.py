def enable_backups(self, table_name, model):
        """Calls UpdateContinuousBackups on the table according to model.Meta["continuous_backups"]

        :param table_name: The name of the table to enable Continuous Backups on
        :param model: The model to get Continuous Backups settings from
        """
        self._tables.pop(table_name, None)
        request = {
            "TableName": table_name,
            "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True}
        }
        try:
            self.dynamodb_client.update_continuous_backups(**request)
        except botocore.exceptions.ClientError as error:
            raise BloopException("Unexpected error while setting Continuous Backups.") from error