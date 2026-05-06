def create_config(self, env, conf):
        """
        Set conf to env under service.

        pass None to env for root.
        """

        if not isinstance(conf, collections.Mapping):
            raise ValueError("conf must be a collections.Mapping")

        self.zk.ensure_path(self.view_path)

        self._create(
            self._get_env_path(env),
            conf
        )

        self._update_view(env)