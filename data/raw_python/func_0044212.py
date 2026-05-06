def _set_remote(self, stream=False):
        """
        Call :py:meth:`~._args_for_remote`; if the return value is not None,
        execute 'terraform remote config' with those arguments and ensure it
        exits 0.

        :param stream: whether or not to stream TF output in realtime
        :type stream: bool
        """
        args = self._args_for_remote()
        if args is None:
            logger.debug('_args_for_remote() returned None; not configuring '
                         'terraform remote')
            return
        logger.warning('Setting terraform remote config: %s', ' '.join(args))
        args = ['config'] + args
        self._run_tf('remote', cmd_args=args, stream=stream)
        logger.info('Terraform remote configured.')