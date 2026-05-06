def restart(self):
        """Restart the debugger after source code changes."""
        _module_finder.reset()
        linecache.checkcache()
        for module_bpts in self.breakpoints.values():
            module_bpts.reset()