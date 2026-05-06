def _flatten_from_root(self, env):
        """
        Flatten values from root down in to new view.
        """

        nodes = env.components

        # Path through the znode graph from root ('') to env
        path = [nodes[:n] for n in xrange(len(nodes) + 1)]

        # Expand path and map it to the root
        path = map(
            self._get_env_path,
            [Env('/'.join(p)) for p in path]
        )

        data = {}
        for n in path:
            _, config = self._get(n)
            data.update(config)

        return data