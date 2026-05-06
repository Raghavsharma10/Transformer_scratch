def channels(self):
        """The number of channels in the audio (an int)."""
        if hasattr(self.mgfile.info, 'channels'):
            return self.mgfile.info.channels
        return 0