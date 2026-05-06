def update(self, event_or_list):
        """Update the button with the events."""

        for e in super().update(event_or_list):
            if e.type == MOUSEBUTTONDOWN:
                if e.pos in self:
                    self.click()
                else:
                    self.release(force_no_call=True)

            elif e.type == MOUSEBUTTONUP:
                self.release(force_no_call=e.pos not in self)

            elif e.type == MOUSEMOTION:
                if e.pos in self:
                    self.hovered = True
                else:
                    self.hovered = False