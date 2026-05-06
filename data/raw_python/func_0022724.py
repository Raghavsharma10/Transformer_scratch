def _update_clipper(self):
        """Called whenever the clipper for this widget may need to be updated.
        """
        if self.clip_children and self._clipper is None:
            self._clipper = Clipper()
        elif not self.clip_children:
            self._clipper = None

        if self._clipper is None:
            return
        self._clipper.rect = self.inner_rect
        self._clipper.transform = self.get_transform('framebuffer', 'visual')