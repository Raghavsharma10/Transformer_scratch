def gui():
    """Main function"""

    global SCREEN_SIZE

    # #######
    # setup all objects
    # #######

    os.environ['SDL_VIDEO_CENTERED'] = '1'  # centers the windows

    screen = new_screen()
    pygame.display.set_caption('Empty project')
    pygame.event.set_allowed([QUIT, KEYDOWN, MOUSEBUTTONDOWN])
    clock = pygame.time.Clock()
    fps = FPSIndicator(clock)

    while True:

        # #######
        # Input loop
        # #######

        # mouse = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == QUIT:
                return 0

            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    return 0

                if e.key == K_F4 and e.mod & KMOD_ALT:  # Alt+F4 --> quits
                    return 0

            if e.type == VIDEORESIZE:
                SCREEN_SIZE = e.size
                screen = new_screen()

        # #######
        # Draw all
        # #######

        screen.fill(WHITE)
        fps.render(screen)

        pygame.display.update()
        clock.tick(FPS)