def set_scissor(self, x, y, w, h):
        """Define the scissor box
    
        Parameters
        ----------
        x : int
            Left corner of the box.
        y : int
            Lower corner of the box.
        w : int
            The width of the box.
        h : int
            The height of the box.
        """
        self.glir.command('FUNC', 'glScissor', int(x), int(y), int(w), int(h))