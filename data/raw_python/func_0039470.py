def composed_deps(self):
        """Dependencies of this build target."""
        if 'deps' in self.params:
            param_deps = self.params['deps'] or []
            deps = [self.makeaddress(dep) for dep in param_deps]
            return deps
        else:
            return None