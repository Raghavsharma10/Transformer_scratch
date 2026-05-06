def compile(self, pretty=True):
        """ Compile all code and return a dict {name: code} where the keys
        are determined by the keyword arguments passed to __init__().

        Parameters
        ----------
        pretty : bool
            If True, use a slower method to mangle object names. This produces
            GLSL that is more readable.
            If False, then the output is mostly unreadable GLSL, but is about
            10x faster to compile.

        """
        # Authoritative mapping of {obj: name}
        self._object_names = {}

        #
        # 1. collect list of dependencies for each shader
        #

        # maps {shader_name: [deps]}
        self._shader_deps = {}

        for shader_name, shader in self.shaders.items():
            this_shader_deps = []
            self._shader_deps[shader_name] = this_shader_deps
            dep_set = set()

            for dep in shader.dependencies(sort=True):
                # visit each object no more than once per shader
                if dep.name is None or dep in dep_set:
                    continue
                this_shader_deps.append(dep)
                dep_set.add(dep)

        #
        # 2. Assign names to all objects.
        #
        if pretty:
            self._rename_objects_pretty()
        else:
            self._rename_objects_fast()

        #
        # 3. Now we have a complete namespace; concatenate all definitions
        # together in topological order.
        #
        compiled = {}
        obj_names = self._object_names

        for shader_name, shader in self.shaders.items():
            code = []
            for dep in self._shader_deps[shader_name]:
                dep_code = dep.definition(obj_names)
                if dep_code is not None:
                    # strip out version pragma if present;
                    regex = r'#version (\d+)'
                    m = re.search(regex, dep_code)
                    if m is not None:
                        # check requested version
                        if m.group(1) != '120':
                            raise RuntimeError("Currently only GLSL #version "
                                               "120 is supported.")
                        dep_code = re.sub(regex, '', dep_code)
                    code.append(dep_code)

            compiled[shader_name] = '\n'.join(code)

        self.code = compiled
        return compiled