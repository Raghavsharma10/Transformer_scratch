def play_async(cls, file_path, on_done=None):
        """ Play an audio file asynchronously.

        :param file_path: the path to the file to play.
        :param on_done: callback when audio playback completes.
        """
        thread = threading.Thread(
            target=AudioPlayer.play, args=(file_path, on_done,))
        thread.start()