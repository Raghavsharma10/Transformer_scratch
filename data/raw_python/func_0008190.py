def scale2x(self, surface):
        """
        Scales using the AdvanceMAME Scale2X algorithm which does a
        'jaggie-less' scale of bitmap graphics.
        """
        assert(self._scale == 2)
        return self._pygame.transform.scale2x(surface)