def new_screen(self):
        """Makes a new screen with a size of SCREEN_SIZE, and VIDEO_OPTION as flags. Sets the windows name to NAME."""
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.display.set_caption(self.NAME)

        screen_s = self.SCREEN_SIZE
        video_options = self.VIDEO_OPTIONS
        if FULLSCREEN & self.VIDEO_OPTIONS:
            video_options ^= FULLSCREEN
            video_options |= NOFRAME
            screen_s = (0, 0)

        screen = pygame.display.set_mode(screen_s, video_options)

        if FULLSCREEN & self.VIDEO_OPTIONS:
            self.SCREEN_SIZE = screen.get_size()

        if not QUIT in self.EVENT_ALLOWED:
            self.EVENT_ALLOWED = list(self.EVENT_ALLOWED)
            self.EVENT_ALLOWED.append(QUIT)

        pygame.event.set_allowed(self.EVENT_ALLOWED)

        return screen