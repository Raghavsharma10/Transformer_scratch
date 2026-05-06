def _setup_tf(self, stream=False):
        """
        Setup terraform; either 'remote config' or 'init' depending on version.
        """
        if self.tf_version < (0, 9, 0):
            self._set_remote(stream=stream)
            return
        self._run_tf('init', stream=stream)
        logger.info('Terraform initialized')