def _set(self, data, version):
        """serialize and set data to self.path."""

        self.zk.set(self.path, json.dumps(data), version)