def smoothscale(self, surface):
        """
        Smooth scaling using MMX or SSE extensions if available
        """
        return self._pygame.transform.smoothscale(surface, self._output_size)