def display_console(self, request):
        """Display a standalone shell."""
        if 0 not in self.frames:
            self.frames[0] = _ConsoleFrame(self.console_init_func())
        return Response(render_console_html(secret=self.secret),
            content_type='text/html')