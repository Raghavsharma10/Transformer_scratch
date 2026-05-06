def _add_conversion(self, plugin, pbt):
        """Add a new PluginBlockType conversion.

        If the plugin already exists, do nothing.

        """
        assert self.shape == pbt.shape
        assert len(self.inserts) == len(pbt.inserts)
        for (i, o) in zip(self.inserts, pbt.inserts):
            assert i.shape == o.shape
            assert i.kind == o.kind
            assert i.unevaluated == o.unevaluated
        if plugin not in self._plugins:
            self._plugins[plugin] = pbt