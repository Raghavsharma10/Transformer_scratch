def ctor_args(self):
        """Return arguments for constructing a copy"""

        return dict(
            config=self._config,
            search=self._search,
            echo=self._echo,
            read_only=self.read_only
        )