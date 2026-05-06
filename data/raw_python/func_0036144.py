def set_config(self, env, conf, version):
        """
        Set conf to env under service.

        pass None to env for root.
        """

        if not isinstance(conf, collections.Mapping):
            raise ValueError("conf must be a collections.Mapping")

        self._set(
            self._get_env_path(env),
            conf,
            version
        )
        path = self._get_env_path(env)
        """Update env's children with new config."""
        for child in zkutil.walk(self.zk, path):
            self._update_view(Env(child[len(self.conf_path)+1:]))