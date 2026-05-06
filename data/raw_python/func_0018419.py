def gui():
    """Main function"""

    # #######
    # setup all objects
    # #######
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREEN_SIZE, DOUBLEBUF | NOFRAME)
    pygame.event.set_allowed([QUIT, KEYDOWN, MOUSEBUTTONDOWN])

    game = Morpion()

    run = True
    while run:

        # #######
        # Input loop
        # #######

        mouse = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == QUIT:
                run = False

            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    run = False

                if e.key == K_F4 and e.mod & KMOD_ALT:
                    return 0

            elif e.type == MOUSEBUTTONDOWN:
                if e.button == 1:
                    if pos_from_mouse(mouse):

                        if game.is_full() or game.is_won():
                            game = Morpion()
                            continue

                        x, y = pos_from_mouse(mouse)

                        try:
                            game.play(x, y)
                        except IndexError:
                            pass

        if pos_from_mouse(mouse):
            x, y = pos_from_mouse(mouse)
            game.hint(x, y)

        # #######
        # Draw all
        # #######

        screen.fill(WHITE)
        game.render(screen)

        pygame.display.update()
        clock.tick(FPS)