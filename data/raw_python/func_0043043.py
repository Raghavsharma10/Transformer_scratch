def rebuild(self):
        """Rebuild RIFF tree and index from streams."""
        movi = self.riff.find('LIST', 'movi')
        movi.chunks = self.combine_streams()
        self.rebuild_index()