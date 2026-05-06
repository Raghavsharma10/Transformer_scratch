def draw(self):
        """Draw the visual
        """
        if not self.visible:
            return
        if self._prepare_draw(view=self) is False:
            return

        for v in self._subvisuals:
            if v.visible:
                v.draw()