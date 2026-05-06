def _args_for_remote(self):
        """
        Generate arguments for 'terraform remote config'. Return None if
        not present in configuration.

        :return: list of args for 'terraform remote config' or None
        :rtype: :std:term:`list`
        """
        conf = self.config.get('terraform_remote_state')
        if conf is None:
            return None
        args = ['-backend=%s' % conf['backend']]
        for k, v in sorted(conf['config'].items()):
            args.append('-backend-config="%s=%s"' % (k, v))
        return args