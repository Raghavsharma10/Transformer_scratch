def play(cls, file_path, on_done=None, logger=None):
        """ Play an audio file.

        :param file_path: the path to the file to play.
        :param on_done: callback when audio playback completes.
        """
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(file_path)
        except pygame.error as e:
            if logger is not None:
                logger.warning(str(e))
            return

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            continue
        if on_done:
            on_done()