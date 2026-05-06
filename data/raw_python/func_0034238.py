def draw(self):
        """ Draw the virtual keyboard into the delegate surface object if enabled. """
        if self.state > 0:
            self.renderer.draw_background(self.surface, self.layout.position, self.layout.size)
            for row in self.layout.rows:
                for key in row.keys:
                    self.renderer.draw_key(self.surface, key)