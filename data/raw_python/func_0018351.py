def update_screen(self):
        """Refresh the screen. You don't need to override this except to update only small portins of the screen."""
        self.clock.tick(self.FPS)
        pygame.display.update()