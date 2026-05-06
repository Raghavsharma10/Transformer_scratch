def flush(self):
        """Flush GL commands
    
        This is a wrapper for glFlush(). This also flushes the GLIR
        command queue.
        """
        if hasattr(self, 'flush_commands'):
            context = self
        else:
            context = get_current_canvas().context
        context.glir.command('FUNC', 'glFlush')
        context.flush_commands()