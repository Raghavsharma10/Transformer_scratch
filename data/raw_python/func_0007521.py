def shrink_wrap(self):
        """Tightly bound the current text respecting current padding."""

        self.frame.size = (self.text_size[0] + self.padding[0] * 2,
                           self.text_size[1] + self.padding[1] * 2)