def flush_commands(self, event=None):
        """ Flush

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        if self._do_CURRENT_command:
            self._do_CURRENT_command = False
            canvas = get_current_canvas()
            if canvas and hasattr(canvas, '_backend'):
                fbo = canvas._backend._vispy_get_fb_bind_location()
            else:
                fbo = 0
            self.shared.parser.parse([('CURRENT', 0, fbo)])
        self.glir.flush(self.shared.parser)