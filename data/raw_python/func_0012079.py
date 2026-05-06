def display(surface):
    """Displays a pygame.Surface in the window.
    
    in pygame the window is represented through a surface, on which you can draw
    as on any other pygame.Surface. A refernce to to the screen can be optained
    via the :py:func:`pygame.display.get_surface` function. To display the
    contents of the screen surface in the window :py:func:`pygame.display.flip`
    needs to be called.
    
    :py:func:`display` draws the surface onto the screen surface at the postion
    (0, 0), and then calls :py:func:`flip`.
    
    :param surface: the pygame.Surface to display
    :type surface: pygame.Surface
    """
    screen = pygame.display.get_surface()
    screen.blit(surface, (0, 0))
    pygame.display.flip()