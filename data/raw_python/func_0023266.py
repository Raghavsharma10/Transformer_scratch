def _rename_objects_fast(self):
        """ Rename all objects quickly to guaranteed-unique names using the
        id() of each object.

        This produces mostly unreadable GLSL, but is about 10x faster to
        compile.
        """
        for shader_name, deps in self._shader_deps.items():
            for dep in deps:
                name = dep.name
                if name != 'main':
                    ext = '_%x' % id(dep)
                    name = name[:32-len(ext)] + ext
                self._object_names[dep] = name