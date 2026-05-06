def resize(self, signum, obj):
        """ handler for SIGWINCH """
        self.s.clear()
        stream_cursor = self.pads['streams'].getyx()[0]
        for pad in self.pads.values():
            pad.clear()
        self.s.refresh()
        self.set_screen_size()
        self.set_title(TITLE_STRING)
        self.init_help()
        self.init_streams_pad()
        self.move(stream_cursor, absolute=True, pad_name='streams', refresh=False)
        self.s.refresh()
        self.show()