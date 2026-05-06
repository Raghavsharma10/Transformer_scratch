def _update(self, archive_name, version_metadata):
        '''
        Updates the version specific metadata attribute in DynamoDB
        In DynamoDB this is simply a list append on this attribute value

        Parameters
        ----------
        archive_name: str
            unique '_id' primary key

        version_metadata: dict
            dictionary of version metadata values

        Returns
        -------
        dict
            list of dictionaries of version_history
        '''

        command = "SET version_history = list_append(version_history, :v)"

        self._table.update_item(
            Key={'_id': archive_name},
            UpdateExpression=command,
            ExpressionAttributeValues={':v': [version_metadata]},
            ReturnValues='ALL_NEW')