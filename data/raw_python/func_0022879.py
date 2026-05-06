def clear(self, color=True, depth=True, stencil=True):
        """Clear the screen buffers
    
        This is a wrapper for gl.glClear.
    
        Parameters
        ----------
        color : bool | str | tuple | instance of Color
            Clear the color buffer bit. If not bool, ``set_clear_color`` will
            be used to set the color clear value.
        depth : bool | float
            Clear the depth buffer bit. If float, ``set_clear_depth`` will
            be used to set the depth clear value.
        stencil : bool | int
            Clear the stencil buffer bit. If int, ``set_clear_stencil`` will
            be used to set the stencil clear index.
        """
        bits = 0
        if isinstance(color, np.ndarray) or bool(color):
            if not isinstance(color, bool):
                self.set_clear_color(color)
            bits |= gl.GL_COLOR_BUFFER_BIT
        if depth:
            if not isinstance(depth, bool):
                self.set_clear_depth(depth)
            bits |= gl.GL_DEPTH_BUFFER_BIT
        if stencil:
            if not isinstance(stencil, bool):
                self.set_clear_stencil(stencil)
            bits |= gl.GL_STENCIL_BUFFER_BIT
        self.glir.command('FUNC', 'glClear', bits)