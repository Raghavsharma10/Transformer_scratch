def remove_watchers(self, watcher_name = None):
        """
        Remove watcher with *watcher_name* or remove all watchers.
        """
        if watcher_name == None:
            self.device.watchers.remove()
        else:
            self.device.watchers.remove(watcher_name)