def identity(self, surface):
        """
        Fast scale operation that does not sample the results
        """
        return self._pygame.transform.scale(surface, self._output_size)