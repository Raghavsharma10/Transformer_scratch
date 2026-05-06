def set_shaders(self, vert, frag):
        """ Set the vertex and fragment shaders.
        
        Parameters
        ----------
        vert : str
            Source code for vertex shader.
        frag : str
            Source code for fragment shaders.
        """
        if not vert or not frag:
            raise ValueError('Vertex and fragment code must both be non-empty')
        
        # pre-process shader code for #include directives
        vert, frag = preprocess(vert), preprocess(frag)
        
        # Store source code, send it to glir, parse the code for variables
        self._shaders = vert, frag

        self._glir.command('SHADERS', self._id, vert, frag)
        # All current variables become pending variables again
        for key, val in self._user_variables.items():
            self._pending_variables[key] = val
        self._user_variables = {}
        # Parse code (and process pending variables)
        self._parse_variables_from_code()