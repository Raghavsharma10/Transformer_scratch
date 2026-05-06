def render(self, path, width = 1024, height = 1024):
    """ Renders the current page to a PNG file (viewport size in pixels). """
    self.conn.issue_command("Render", path, width, height)