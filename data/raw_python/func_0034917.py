def initialize_shade(self, shade_name, shade_color, alpha):
        """This method will create semi-transparent surfaces with a specified
        color. The surface can be toggled on and off.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Inputs:
            Shade_name - String of the name that you want to associate with the
                surface

            Shade_color - An rgb tuple of the color of the shade

            Alpha - Level of transparency of the shade (0-255 with 150 being a
                good middle value)

        (doc string updated ver 0.1)
        """

        # Create the pygame surface
        self.shades[shade_name] = [0, pygame.Surface(self.image.get_size())]

        # Fill the surface with a solid color or an image
        if type(shade_color) == str:
            background = pygame.image.load(shade_color).convert()
            background = pygame.transform.scale(background,
                                                (self.image.get_width(),
                                                 self.image.get_height()))
            self.shades[shade_name][1].blit(background, (0, 0))
        # Otherwise the background should contain an rgb value
        else:
            self.shades[shade_name][1].fill(shade_color)

        # Set the alpha value for the shade
        self.shades[shade_name][1].set_alpha(alpha)