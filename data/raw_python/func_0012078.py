def init(resolution, pygame_flags=0, display_pos=(0, 0), interactive_mode=False):
    """Creates a window of given resolution.

    :param resolution: the resolution of the windows as (width, height) in
        pixels
    :type resolution: tuple
    :param pygame_flags: modify the creation of the window.
        For further information see :ref:`creating_a_window`
    :type pygame_flags: int
    :param display_pos: determines the position on the desktop where the
        window is created. In a multi monitor system this can be used to position
        the window on a different monitor. E.g. the monitor to the right of the
        main-monitor would be at position (1920, 0) if the main monitor has the
        width 1920.
    :type display_pos: tuple
    :param interactive_mode: Will install a thread, that emptys the
        event-queue every 100ms. This is neccessary to be able to use the
        display() function in an interactive console on windows systems.
        If interactive_mode is set, init() will return a reference to the
        background thread. This thread has a stop() method which can be used to
        cancel it. If you use ctrl+d or exit() within ipython, while the thread
        is still running, ipython will become unusable, but not close. 
    :type interactive_mode: bool

    :return: a reference to the display screen, or a reference to the background
        thread if interactive_mode was set to true. In the second scenario you
        can obtain a reference to the display surface via
        pygame.display.get_surface()
        
    :rtype: pygame.Surface
    """

    os.environ['SDL_VIDEO_WINDOW_POS'] = "{}, {}".format(*display_pos)
    pygame.init()
    pygame.font.init()
    disp = pygame.display.set_mode(resolution, pygame_flags)
    return _PumpThread() if interactive_mode else disp