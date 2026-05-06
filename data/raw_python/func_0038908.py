def mount(self, readonly=True):
        """Mounts the given disk.
        It must be called before any other method.

        """
        self._handler.add_drive_opts(self.disk_path, readonly=True)
        self._handler.launch()

        for mountpoint, device in self._inspect_disk():
            if readonly:
                self._handler.mount_ro(device, mountpoint)
            else:
                self._handler.mount(device, mountpoint)

        if self._handler.inspect_get_type(self._root) == 'windows':
            self.path = self._windows_path
        else:
            self.path = posix_path