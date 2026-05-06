def _inspect_disk(self):
        """Inspects the disk and returns the mountpoints mapping
        as a list which order is the supposed one for correct mounting.

        """
        roots = self._handler.inspect_os()

        if roots:
            self._root = roots[0]
            return sorted(self._handler.inspect_get_mountpoints(self._root),
                          key=lambda m: len(m[0]))
        else:
            raise RuntimeError("No OS found on the given disk image.")