def _pfp__watch(self, watcher):
        """Add the watcher to the list of fields that
        are watching this field
        """
        if self._pfp__parent is not None and isinstance(self._pfp__parent, Union):
            self._pfp__parent._pfp__watch(watcher)
        else:
            self._pfp__watchers.append(watcher)