def glir(self):
        """ The GLIR queue corresponding to the current canvas
        """
        canvas = get_current_canvas()
        if canvas is None:
            msg = ("If you want to use gloo without vispy.app, " + 
                   "use a gloo.context.FakeCanvas.")
            raise RuntimeError('Gloo requires a Canvas to run.\n' + msg)
        return canvas.context.glir