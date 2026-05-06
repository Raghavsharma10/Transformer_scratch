def load_file(self, element=None, filename=None, clear_registry=True):
        """Return the namespace of the given file (and eventually of its
        corresponding auxiliary subfiles) as a |dict|.

        By default, the internal registry is cleared when a control file and
        all its corresponding auxiliary files have been loaded.  You can
        change this behaviour by passing `False` for the `clear_registry`
        argument.  This might decrease model initialization times
        significantly.  But then it is your own responsibility to call
        method |ControlManager.clear_registry| when necessary (before
        reloading a changed control file).
        """
        if not filename:
            filename = element.name
        type(self)._workingpath = self.currentpath
        info = {}
        if element:
            info['element'] = element
        try:
            self.read2dict(filename, info)
        finally:
            type(self)._workingpath = '.'
            if clear_registry:
                self._registry.clear()
        return info