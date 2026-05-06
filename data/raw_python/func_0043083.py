def _parse_env(self):
        """Parse the environment variables for any configuration if an `env_prefix`
        is set.
        """
        env_cfg = DotDict()

        # if the env prefix doesn't end with '_', we'll append it here
        if self.env_prefix and not self.env_prefix.endswith('_'):
            self.env_prefix = self.env_prefix + '_'

        # if there is no scheme, we won't know what to look for so only parse
        # config if there is a scheme.
        if self.scheme:
            for k, v in self.scheme.flatten().items():
                value = v.parse_env(k, self.env_prefix, self.auto_env)
                if value is not None:
                    env_cfg[k] = value

        if len(env_cfg) > 0:
            # the configuration changes, so we invalidate the cached config
            self._full_config = None
            self._environment.update(env_cfg)