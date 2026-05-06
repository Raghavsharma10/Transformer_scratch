def _rename_objects_pretty(self):
        """ Rename all objects like "name_1" to avoid conflicts. Objects are
        only renamed if necessary.

        This method produces more readable GLSL, but is rather slow.
        """
        #
        # 1. For each object, add its static names to the global namespace
        #    and make a list of the shaders used by the object.
        #

        # {name: obj} mapping for finding unique names
        # initialize with reserved keywords.
        self._global_ns = dict([(kwd, None) for kwd in gloo.util.KEYWORDS])
        # functions are local per-shader
        self._shader_ns = dict([(shader, {}) for shader in self.shaders])

        # for each object, keep a list of shaders the object appears in
        obj_shaders = {}

        for shader_name, deps in self._shader_deps.items():
            for dep in deps:
                # Add static names to namespace
                for name in dep.static_names():
                    self._global_ns[name] = None

                obj_shaders.setdefault(dep, []).append(shader_name)

        #
        # 2. Assign new object names
        #
        name_index = {}
        for obj, shaders in obj_shaders.items():
            name = obj.name
            if self._name_available(obj, name, shaders):
                # hooray, we get to keep this name
                self._assign_name(obj, name, shaders)
            else:
                # boo, find a new name
                while True:
                    index = name_index.get(name, 0) + 1
                    name_index[name] = index
                    ext = '_%d' % index
                    new_name = name[:32-len(ext)] + ext
                    if self._name_available(obj, new_name, shaders):
                        self._assign_name(obj, new_name, shaders)
                        break