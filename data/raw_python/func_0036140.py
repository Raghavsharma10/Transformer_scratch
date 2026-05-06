def _get(self):
        """get and parse data stored in self.path."""

        data, stat = self.zk.get(self.path)
        if not len(data):
            return {}, stat.version
        if self.OLD_SEPARATOR in data:
            return self._get_old()
        return json.loads(data), stat.version