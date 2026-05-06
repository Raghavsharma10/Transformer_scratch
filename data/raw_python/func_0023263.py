def glsl_type(self):
        """ GLSL declaration strings required for a variable to hold this data.
        """
        if self.dtype is None:
            return None
        dtshape = self.dtype[0].shape
        n = dtshape[0] if dtshape else 1
        if n > 1:
            dtype = 'vec%d' % n
        else:
            dtype = 'float' if 'f' in self.dtype[0].base.kind else 'int'
        return 'attribute', dtype