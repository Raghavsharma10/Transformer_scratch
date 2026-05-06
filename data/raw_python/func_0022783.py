def _get_active_attributes_and_uniforms(self):
        """ Retrieve active attributes and uniforms to be able to check that
        all uniforms/attributes are set by the user.
        Other GLIR implementations may omit this.
        """
        # This match a name of the form "name[size]" (= array)
        regex = re.compile("""(?P<name>\w+)\s*(\[(?P<size>\d+)\])\s*""")
        # Get how many active attributes and uniforms there are
        cu = gl.glGetProgramParameter(self._handle, gl.GL_ACTIVE_UNIFORMS)
        ca = gl.glGetProgramParameter(self.handle, gl.GL_ACTIVE_ATTRIBUTES)
        # Get info on each one
        attributes = []
        uniforms = []
        for container, count, func in [(attributes, ca, gl.glGetActiveAttrib),
                                       (uniforms, cu, gl.glGetActiveUniform)]:
            for i in range(count):
                name, size, gtype = func(self._handle, i)
                m = regex.match(name)  # Check if xxx[0] instead of xx
                if m:
                    name = m.group('name')
                    for i in range(size):
                        container.append(('%s[%d]' % (name, i), gtype))
                else:
                    container.append((name, gtype))
        #return attributes, uniforms
        return set([v[0] for v in attributes] + [v[0] for v in uniforms])