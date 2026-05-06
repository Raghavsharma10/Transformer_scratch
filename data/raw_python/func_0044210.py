def _validate(self):
        """
        Confirm that we can run terraform (by calling its version action)
        and then validate the configuration.
        """
        try:
            out = self._run_tf('version')
        except:
            raise Exception('ERROR: executing \'%s version\' failed; is '
                            'terraform installed and is the path to it (%s) '
                            'correct?' % (self.tf_path, self.tf_path))
        res = re.search(r'Terraform v(\d+)\.(\d+)\.(\d+)', out)
        if res is None:
            logger.error('Unable to determine terraform version; will not '
                         'validate config. Note that this may cause problems '
                         'when using older Terraform versions. This program '
                         'requires Terraform >= 0.6.16.')
            return
        self.tf_version = (
            int(res.group(1)), int(res.group(2)), int(res.group(3))
        )
        logger.debug('Terraform version: %s', self.tf_version)
        if self.tf_version < (0, 6, 16):
            raise Exception('This program requires Terraform >= 0.6.16, as '
                            'that version introduces a bug fix for working '
                            'with api_gateway_integration_response resources; '
                            'see: https://github.com/hashicorp/terraform/pull'
                            '/5893')
        try:
            self._run_tf('validate', ['.'])
        except Exception as ex:
            logger.critical("Terraform config validation failed. "
                            "This is almost certainly a bug in "
                            "webhook2lambda2sqs; please re-run with '-vv' and "
                            "open a bug at <https://github.com/jantman/"
                            "webhook2lambda2sqs/issues>. Exception: %s", ex)
            raise Exception(
                'ERROR: Terraform config validation failed: %s' % ex
            )