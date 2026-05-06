def set_method_settings(self):
        """
        Set the Method settings <https://docs.aws.amazon.com/apigateway/api-\
reference/resource/stage/#methodSettings> on our Deployment Stage.
        This is currently not supported by Terraform; see <https://github.com/\
jantman/webhook2lambda2sqs/issues/7> and <https://github.com/hashicorp\
/terraform/issues/6612>.

        Calls :py:meth:`~._add_method_setting` for each setting that is not
        currently correct.
        """
        settings = self.config.get('api_gateway_method_settings')
        if settings is None:
            logger.debug('api_gateway_method_settings not set in config')
            return
        logger.info('Setting API Gateway Stage methodSettings')
        api_id = self.get_api_id()
        stage_name = self.config.stage_name
        logger.debug('Connecting to AWS apigateway API')
        conn = client('apigateway')
        logger.debug('Getting Stage configuration: api_id=%s stage_name=%s',
                     api_id, stage_name)
        stage = conn.get_stage(restApiId=api_id, stageName=stage_name)
        logger.debug("Got stage config: \n%s", pformat(stage))
        # hack for stages that have had no method settings applied yet
        if '*/*' not in stage['methodSettings']:
            stage['methodSettings']['*/*'] = {}
        curr_settings = stage['methodSettings']['*/*']
        for k, v in sorted(settings.items()):
            if k in curr_settings and curr_settings[k] == v:
                logger.debug('methodSetting "%s" is correct (%s)', k, v)
                continue
            # else update the value; note that the API doesn't actually follow
            # https://tools.ietf.org/html/rfc6902#section-4 and doesn't seem
            # to actually accept 'add' for these.
            op = 'replace'
            if k not in curr_settings:
                logger.debug('Adding new methodSetting "%s" value %s', k, v)
            else:
                logger.debug('Updating methodSetting "%s" from %s to %s',
                             k, curr_settings[k], v)
            self._add_method_setting(conn, api_id, stage_name,
                                     self._method_setting_paths[k] % '*/*',
                                     k, v, op)