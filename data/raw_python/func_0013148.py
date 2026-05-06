def _get_env_data(self, reload=False):
        """Get the data about the available environments.

        env_data is a structure {name -> (resourcedir, kernel spec)}
        """

        # This is called much too often and finding-process is really expensive :-(
        if not reload and getattr(self, "_env_data_cache", {}):
            return getattr(self, "_env_data_cache")

        env_data = {}
        for supplyer in ENV_SUPPLYER:
            env_data.update(supplyer(self))

        env_data = {name: env_data[name] for name in env_data if self.validate_env(name)}
        new_kernels = [env for env in list(env_data.keys()) if env not in list(self._env_data_cache.keys())]
        if new_kernels:
            self.log.info("Found new kernels in environments: %s", ", ".join(new_kernels))

        self._env_data_cache = env_data
        return env_data