def reset(cls):
        """
        Reset to default settings
        """
        cls.debug = False
        cls.disabled = False
        cls.overwrite = False
        cls.playback_only = False
        cls.recv_timeout = 5
        cls.recv_endmarkers = []
        cls.recv_size = None