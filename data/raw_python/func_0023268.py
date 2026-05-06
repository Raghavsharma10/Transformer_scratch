def _name_available(self, obj, name, shaders):
        """ Return True if *name* is available for *obj* in *shaders*.
        """
        if name in self._global_ns:
            return False
        shaders = self.shaders if self._is_global(obj) else shaders
        for shader in shaders:
            if name in self._shader_ns[shader]:
                return False
        return True