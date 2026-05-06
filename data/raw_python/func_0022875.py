def set_viewport(self, *args):
        """Set the OpenGL viewport
    
        This is a wrapper for gl.glViewport.
    
        Parameters
        ----------
        *args : tuple
            X and Y coordinates, plus width and height. Can be passed in as
            individual components, or as a single tuple with four values.
        """
        x, y, w, h = args[0] if len(args) == 1 else args
        self.glir.command('FUNC', 'glViewport', int(x), int(y), int(w), int(h))