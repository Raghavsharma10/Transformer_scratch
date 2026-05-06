def reset(self):
        """Remove from sys.modules the modules imported by the debuggee."""
        if not self.hooked:
            self.hooked = True
            sys.path_hooks.append(self)
            sys.path.insert(0, self.PATH_ENTRY)
            return

        for modname in self:
            if modname in sys.modules:
                del sys.modules[modname]
                submods = []
                for subm in sys.modules:
                    if subm.startswith(modname + '.'):
                        submods.append(subm)
                # All submodules of modname may not have been imported by the
                # debuggee, but they are still removed from sys.modules as
                # there is no way to distinguish them.
                for subm in submods:
                    del sys.modules[subm]
        self[:] = []