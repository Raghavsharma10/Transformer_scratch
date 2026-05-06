def restore(self):
        """Restores the modules that the saver knows about into
        sys.modules.
        """
        try:
            for modname, mod in self._saved.items():
                if mod is not None:
                    sys.modules[modname] = mod
                else:
                    try:
                        del sys.modules[modname]
                    except KeyError:
                        pass
        finally:
            imp.release_lock()