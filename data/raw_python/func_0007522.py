def layout(self):
        """Call to have the view layout itself.

        Subclasses should invoke this after laying out child
        views and/or updating its own frame.
        """
        if self.shadowed:
            shadow_size = theme.current.shadow_size
            shadowed_frame_size = (self.frame.w + shadow_size,
                                   self.frame.h + shadow_size)
            self.surface = pygame.Surface(
                shadowed_frame_size, pygame.SRCALPHA, 32)
            shadow_image = resource.get_image('shadow')
            self.shadow_image = resource.scale_image(shadow_image,
                                                     shadowed_frame_size)
        else:
            self.surface = pygame.Surface(self.frame.size, pygame.SRCALPHA, 32)
            self.shadow_image = None