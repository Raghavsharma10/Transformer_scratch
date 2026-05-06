def _get_old(self):
        """get and parse data stored in self.path."""

        def _deserialize(d):
            if not len(d):
                return {}
            return dict(l.split(self.OLD_SEPARATOR) for l in d.split('\n'))

        data, stat = self.zk.get(self.path)
        return _deserialize(data.decode('utf8')), stat.version