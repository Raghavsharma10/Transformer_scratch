def _assign_name(self, obj, name, shaders):
        """ Assign *name* to *obj* in *shaders*.
        """
        if self._is_global(obj):
            assert name not in self._global_ns
            self._global_ns[name] = obj
        else:
            for shader in shaders:
                ns = self._shader_ns[shader]
                assert name not in ns
                ns[name] = obj
        self._object_names[obj] = name