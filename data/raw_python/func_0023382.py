def draw_text(self, ax, text, force_trans=None, text_type=None):
        """Process a matplotlib text object and call renderer.draw_text"""
        content = text.get_text()
        if content:
            transform = text.get_transform()
            position = text.get_position()
            coords, position = self.process_transform(transform, ax,
                                                      position,
                                                      force_trans=force_trans)
            style = utils.get_text_style(text)
            self.renderer.draw_text(text=content, position=position,
                                    coordinates=coords,
                                    text_type=text_type,
                                    style=style, mplobj=text)