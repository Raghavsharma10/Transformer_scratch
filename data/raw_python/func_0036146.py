def get_view_by_env(self, env):
        """
        Returns the view of `env`.

        """
        version, data = self._get(self._get_view_path(env))
        return data