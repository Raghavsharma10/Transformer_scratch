def makeSong(self):
        """Render abstract animation
        """
        self.makeVisualSong()
        self.makeAudibleSong()
        if self.make_video:
            self.makeAnimation()