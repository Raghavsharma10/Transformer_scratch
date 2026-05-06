def _add_method_setting(self, conn, api_id, stage_name, path, key, value,
                            op):
        """
        Update a single method setting on the specified stage. This uses the
        'add' operation to PATCH the resource.

        :param conn: APIGateway API connection
        :type conn: :py:class:`botocore:APIGateway.Client`
        :param api_id: ReST API ID
        :type api_id: str
        :param stage_name: stage name
        :type stage_name: str
        :param path: path to patch (see https://docs.aws.amazon.com/apigateway/\
api-reference/resource/stage/#methodSettings)
        :type path: str
        :param key: the dictionary key this should update
        :type key: str
        :param value: new value to set
        :param op: PATCH operation to perform, 'add' or 'replace'
        :type op: str
        """
        logger.debug('update_stage PATCH %s on %s; value=%s',
                     op, path, str(value))
        res = conn.update_stage(
            restApiId=api_id,
            stageName=stage_name,
            patchOperations=[
                {
                    'op': op,
                    'path': path,
                    'value': str(value)
                }
            ]
        )
        if res['methodSettings']['*/*'][key] != value:
            logger.error('methodSettings PATCH expected to update %s to %s,'
                         'but instead found value as %s', key, value,
                         res['methodSettings']['*/*'][key])
        else:
            logger.info('Successfully updated methodSetting %s to %s',
                        key, value)