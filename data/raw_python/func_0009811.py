def set_level(self, level):
        """Set open level of the curtains.

        Scale is 0-100
        """
        self.set_service_value(
            self.dimmer_service,
            'LoadLevelTarget',
            'newLoadlevelTarget',
            level)

        self.set_cache_value('level', level)