def apply(self, stream=False):
        """
        Run a 'terraform apply'

        :param stream: whether or not to stream TF output in realtime
        :type stream: bool
        """
        self._setup_tf(stream=stream)
        try:
            self._taint_deployment(stream=stream)
        except Exception:
            pass
        args = ['-input=false', '-refresh=true', '.']
        logger.warning('Running terraform apply: %s', ' '.join(args))
        out = self._run_tf('apply', cmd_args=args, stream=stream)
        if stream:
            logger.warning('Terraform apply finished successfully.')
        else:
            logger.warning("Terraform apply finished successfully:\n%s", out)
        self._show_outputs()