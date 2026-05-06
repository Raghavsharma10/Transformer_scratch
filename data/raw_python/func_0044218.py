def destroy(self, stream=False):
        """
        Run a 'terraform destroy'

        :param stream: whether or not to stream TF output in realtime
        :type stream: bool
        """
        self._setup_tf(stream=stream)
        args = ['-refresh=true', '-force', '.']
        logger.warning('Running terraform destroy: %s', ' '.join(args))
        out = self._run_tf('destroy', cmd_args=args, stream=stream)
        if stream:
            logger.warning('Terraform destroy finished successfully.')
        else:
            logger.warning("Terraform destroy finished successfully:\n%s", out)