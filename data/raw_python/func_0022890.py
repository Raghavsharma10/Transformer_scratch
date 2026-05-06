def finish(self):
        """Wait for GL commands to to finish
        
        This creates a GLIR command for glFinish and then processes the
        GLIR commands. If the GLIR interpreter is remote (e.g. WebGL), this
        function will return before GL has finished processing the commands.
        """
        if hasattr(self, 'flush_commands'):
            context = self
        else:
            context = get_current_canvas().context
        context.glir.command('FUNC', 'glFinish')
        context.flush_commands()