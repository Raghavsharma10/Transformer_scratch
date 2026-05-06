def _update_spec_config(self, document_name, spec):
        '''
        Dynamo implementation of project specific metadata spec

        '''
        # add the updated archive_metadata object to Dynamo
        self._spec_table.update_item(
            Key={'_id': '{}'.format(document_name)},
            UpdateExpression="SET config = :v",
            ExpressionAttributeValues={':v': spec},
            ReturnValues='ALL_NEW')